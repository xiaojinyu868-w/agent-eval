"""
评测 Agent - 外呼任务对话模型指令遵循能力评估系统

核心设计：
1. Agent 自主理解指令（不需要机械解析器）
2. Agent 逐项判定 + 逐字证据提取
3. 确定性约束用代码检查（字数、禁用词等）
4. 分数从判定中机械算出
5. 报告 = 证据 + 判定 + 分数
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional
from openai import OpenAI


# ============================================================
# 数据结构
# ============================================================

@dataclass
class CheckItem:
    """单条检查项"""
    dimension: str          # 维度名 (如 "流程遵循度", "约束遵循度")
    item_id: str            # 检查项ID
    description: str        # 检查项描述
    weight: float = 1.0     # 权重
    verdict: Optional[str] = None   # "pass" / "partial" / "fail"
    evidence: Optional[str] = None  # 逐字引用
    reason: Optional[str] = None    # 判定理由
    score: Optional[float] = None   # 0.0 / 0.5 / 1.0


@dataclass
class DialogueTurn:
    """对话轮次"""
    role: str       # "agent" / "user"
    content: str
    turn_id: int = 0


@dataclass
class EvaluationResult:
    """评测结果"""
    instruction_id: str = ""
    checks: list = field(default_factory=list)  # List[CheckItem]
    deterministic_checks: list = field(default_factory=list)
    total_score: float = 0.0
    max_score: float = 0.0
    score_rate: float = 0.0  # total / max
    dimension_scores: dict = field(default_factory=dict)  # dimension -> {score, max, rate}
    report: str = ""


# ============================================================
# LLM 客户端
# ============================================================

class LLMClient:
    """统一的 LLM 调用客户端 - 通过 raw socket 直连绕过代理限制"""

    DEFAULT_HOST = "mmdcadamsminiserverproxy.polaris"
    DEFAULT_PORT = 25340
    DEFAULT_PATH = "/service/20839/v1"
    DEFAULT_MODEL = "kimi_k2d6"
    DEFAULT_HEADERS = {
        "Adams-Platform-User": os.environ.get("ADAMS_PLATFORM_USER", ""),
        "Adams-User-Token": os.environ.get("ADAMS_USER_TOKEN", ""),
        "Adams-Business": os.environ.get("ADAMS_BUSINESS", ""),
    }

    def __init__(self, host: str = None, port: int = None, path: str = None,
                 model: str = None, extra_headers: dict = None):
        self.host = host or self.DEFAULT_HOST
        self.port = port or self.DEFAULT_PORT
        self.path = path or self.DEFAULT_PATH
        self.model = model or self.DEFAULT_MODEL
        self.extra_headers = extra_headers or self.DEFAULT_HEADERS
        # Resolve IPs at init time
        self._ips = self._resolve_ips()

    def _resolve_ips(self) -> list[str]:
        import socket as _socket
        addrs = _socket.getaddrinfo(self.host, self.port, _socket.AF_INET, _socket.SOCK_STREAM)
        return list(set(a[4][0] for a in addrs))

    def chat(self, messages: list, temperature: float = 0.3, max_tokens: int = 4096) -> str:
        import socket as _socket
        import json as _json

        payload = _json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"thinking": False},
        })

        body_path = f"{self.path}/chat/completions"
        http_req = (
            f"POST {body_path} HTTP/1.1\r\n"
            f"Host: {self.host}:{self.port}\r\n"
            f"Content-Type: application/json\r\n"
            f"Connection: close\r\n"
        )
        for k, v in self.extra_headers.items():
            http_req += f"{k}: {v}\r\n"
        http_req += f"Content-Length: {len(payload)}\r\n\r\n{payload}"

        # Try IPs until one works
        last_err = None
        for ip in self._ips:
            s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
            s.settimeout(60)
            try:
                s.connect((ip, self.port))
                s.sendall(http_req.encode())
                resp = b""
                while True:
                    try:
                        chunk = s.recv(8192)
                        if not chunk:
                            break
                        resp += chunk
                    except _socket.timeout:
                        break
                # Parse HTTP response
                decoded = resp.decode("utf-8", errors="replace")
                # Split headers and body
                parts = decoded.split("\r\n\r\n", 1)
                if len(parts) < 2:
                    continue
                body = parts[1]
                # Handle chunked transfer encoding
                if "transfer-encoding: chunked" in parts[0].lower():
                    # De-chunk
                    chunks = []
                    pos = 0
                    while pos < len(body):
                        line_end = body.find("\r\n", pos)
                        if line_end < 0:
                            break
                        size_str = body[pos:line_end].strip()
                        if not size_str:
                            pos = line_end + 2
                            continue
                        try:
                            chunk_size = int(size_str, 16)
                        except ValueError:
                            break
                        if chunk_size == 0:
                            break
                        chunk_start = line_end + 2
                        chunks.append(body[chunk_start:chunk_start + chunk_size])
                        pos = chunk_start + chunk_size + 2
                    body = "".join(chunks)
                data = _json.loads(body)
                content = data["choices"][0]["message"].get("content", "")
                # Kimi K2.6 may put content in reasoning_content if thinking mode
                if not content:
                    reasoning = data["choices"][0]["message"].get("reasoning_content", "")
                    if reasoning:
                        content = reasoning
                return content
            except Exception as e:
                last_err = e
                continue
            finally:
                s.close()
        raise RuntimeError(f"All IPs failed, last error: {last_err}")


# ============================================================
# 确定性约束检查器（不依赖LLM，代码直接算）
# ============================================================

class DeterministicChecker:
    """确定性约束检查 - 字数、禁用词、开场白等"""

    def check_response_length(self, turn: DialogueTurn, max_chars: int) -> CheckItem:
        """检查单条回复字数"""
        char_count = len(turn.content)
        passed = char_count <= max_chars
        return CheckItem(
            dimension="约束遵循度",
            item_id=f"constraint_length_turn_{turn.turn_id}",
            description=f"第{turn.turn_id}轮回复字数不超过{max_chars}字",
            verdict="pass" if passed else "fail",
            evidence=f'"{turn.content[:50]}..."（共{char_count}字）' if char_count > 30 else f'"{turn.content}"（共{char_count}字）',
            reason=f"实际{char_count}字，限制{max_chars}字",
            score=1.0 if passed else 0.0,
        )

    def check_banned_words(self, turn: DialogueTurn, banned: list[str]) -> list[CheckItem]:
        """检查禁用词"""
        items = []
        found = []
        for word in banned:
            if word in turn.content:
                found.append(word)
        passed = len(found) == 0
        items.append(CheckItem(
            dimension="约束遵循度",
            item_id=f"constraint_banned_turn_{turn.turn_id}",
            description=f"第{turn.turn_id}轮不使用禁用词{banned}",
            verdict="pass" if passed else "fail",
            evidence=f"发现禁用词: {found}" if found else "未发现禁用词",
            reason=f"发现{len(found)}个禁用词" if found else "未使用禁用词",
            score=1.0 if passed else 0.0,
        ))
        return items

    def check_opening_line(self, first_agent_turn: DialogueTurn, expected_opening: str) -> CheckItem:
        """检查开场白是否包含关键信息"""
        # 提取 expected_opening 中的关键片段（${...} 是变量，跳过）
        expected_clean = re.sub(r'\$\{[^}]+\}', '', expected_opening)
        expected_words = [w for w in re.split(r'[，。？！、\s]+', expected_clean) if len(w) >= 2]

        matched = []
        for w in expected_words:
            if w in first_agent_turn.content:
                matched.append(w)

        if len(expected_words) == 0:
            rate = 1.0
        else:
            rate = len(matched) / len(expected_words)

        if rate >= 0.6:
            verdict = "pass"
        elif rate >= 0.3:
            verdict = "partial"
        else:
            verdict = "fail"

        return CheckItem(
            dimension="开场白准确度",
            item_id="opening_line_accuracy",
            description="开场白包含指定关键信息",
            verdict=verdict,
            evidence=f'实际开场白: "{first_agent_turn.content[:100]}"',
            reason=f"关键信息匹配率{rate:.0%}，命中{matched}" if matched else f"关键信息匹配率{rate:.0%}，未命中",
            score=rate,
        )


# ============================================================
# 评测 Agent（核心）
# ============================================================

class EvaluationAgent:
    """
    评测 Agent - 自主理解指令，逐项判定，产出可解释报告

    工作流程：
    1. 读取指令 → Agent 自主理解流程、约束、知识点
    2. 生成检查清单 → Agent 根据理解生成结构化 checklist
    3. 逐项判定 → Agent 对每个 checklist 项：找证据 → 做判定
    4. 确定性检查 → 代码直接检查字数、禁用词等
    5. 机械算分 → 从判定结果中算出分数
    6. 生成报告
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.det_checker = DeterministicChecker()

    def evaluate(self, instruction: str, dialogue: list[DialogueTurn], instruction_id: str = "1") -> EvaluationResult:
        """执行完整评测流程"""
        result = EvaluationResult(instruction_id=instruction_id)

        # Step 1: Agent 自主理解指令，生成检查清单
        checklist = self._generate_checklist(instruction)

        # Step 2: 逐项判定（Agent 推理 + 证据提取）
        for item in checklist:
            judged = self._judge_checkitem(item, dialogue, instruction)
            result.checks.append(judged)

        # Step 3: 确定性检查
        det_checks = self._run_deterministic_checks(instruction, dialogue)
        result.deterministic_checks = det_checks

        # Step 4: 机械算分
        all_checks = result.checks + result.deterministic_checks
        self._compute_scores(all_checks, result)

        # Step 5: 生成报告
        result.report = self._generate_report(result, instruction)

        return result

    def _generate_checklist(self, instruction: str) -> list[CheckItem]:
        """Agent 读取指令，自主生成检查清单"""
        prompt = f"""你是一个专业的对话质量评测专家。请仔细阅读以下外呼任务指令，然后生成一个结构化的检查清单（checklist），用于评测对话模型是否正确遵循了该指令。

指令内容：
---
{instruction}
---

请从以下维度生成检查项：
1. **流程遵循度**：是否按指令中的对话流程/步骤执行？每个关键步骤是否覆盖？条件分支是否正确处理？
2. **信息传达完整性**：指令中要求传达的关键信息是否都已传达？数字/参数是否准确？
3. **约束遵循度**：各类约束条件是否遵守？（注意：字数限制、禁用词等确定性约束会由代码单独检查，这里只关注需要语义理解的约束）
4. **任务达成度**：核心任务是否完成？对话是否正常结束？用户问题是否得到回应？

输出JSON格式，每个检查项包含：
- dimension: 维度名
- item_id: 唯一ID（如 flow_step1, info_contract_days, constraint_oot, task_completion）
- description: 检查项描述（要具体，让人一看就知道在检查什么）
- weight: 权重（1.0=普通，1.5=重要，2.0=关键）

只输出JSON数组，不要其他文字。"""

        resp = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.1)
        # 提取JSON
        checklist = self._parse_json_list(resp)
        items = []
        for i, c in enumerate(checklist):
            items.append(CheckItem(
                dimension=c.get("dimension", "未知"),
                item_id=c.get("item_id", f"item_{i}"),
                description=c.get("description", ""),
                weight=c.get("weight", 1.0),
            ))
        return items

    def _judge_checkitem(self, item: CheckItem, dialogue: list[DialogueTurn], instruction: str) -> CheckItem:
        """Agent 对单条检查项做判定：找证据 → 给判定"""
        # 构建对话文本
        dialogue_text = self._format_dialogue(dialogue)

        prompt = f"""你是一个专业的对话质量评测专家。请根据以下对话记录，判定被测Agent是否满足指定的检查项。

【任务指令（参考）】
{instruction}

【对话记录】
{dialogue_text}

【检查项】
- 维度: {item.dimension}
- 检查内容: {item.description}

请按以下格式输出JSON：
{{
  "verdict": "pass" 或 "partial" 或 "fail",
  "evidence": "从对话中逐字引用的关键证据（必须包含具体的轮次和原文）",
  "reason": "简要说明判定理由"
}}

判定标准：
- pass: 完全满足检查项要求
- partial: 部分满足，有欠缺但不严重
- fail: 不满足，有明显遗漏或错误

注意：evidence必须是对话中的原文引用，不能自己编造。如果找不到相关证据，evidence写"未找到相关对话"，verdict写"fail"。

只输出JSON，不要其他文字。"""

        resp = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.1)
        parsed = self._parse_json_single(resp)

        item.verdict = parsed.get("verdict", "fail")
        item.evidence = parsed.get("evidence", "无证据")
        item.reason = parsed.get("reason", "无理由")
        item.score = {"pass": 1.0, "partial": 0.5, "fail": 0.0}.get(item.verdict, 0.0)

        return item

    def _run_deterministic_checks(self, instruction: str, dialogue: list[DialogueTurn]) -> list[CheckItem]:
        """确定性约束检查"""
        checks = []

        # 从指令中提取约束参数
        # 字数限制
        length_match = re.search(r'约\s*(\d+)\s*个?字', instruction)
        length_match2 = re.search(r'最多\s*(\d+)[-~]\s*(\d+)\s*个?字', instruction)
        max_chars = None
        if length_match2:
            max_chars = int(length_match2.group(2))
        elif length_match:
            max_chars = int(length_match.group(1))

        if max_chars:
            for turn in dialogue:
                if turn.role == "agent":
                    checks.append(self.det_checker.check_response_length(turn, max_chars))

        # 禁用词
        banned_patterns = []
        if '不说"好的"' in instruction or '不说"好的"' in instruction or '不说' in instruction:
            banned_extract = re.findall(r'不说["""]([^"""]+)["""]', instruction)
            banned_patterns.extend(banned_extract)
        # 常见禁用语气词
        if '语气词' in instruction or '哈哈' in instruction:
            for w in ['好的', '哈哈', '嘿嘿', '嘻嘻']:
                if w in instruction:
                    banned_patterns.append(w)

        if banned_patterns:
            for turn in dialogue:
                if turn.role == "agent":
                    checks.extend(self.det_checker.check_banned_words(turn, banned_patterns))

        # 开场白
        opening_match = re.search(r'#?\s*Opening Line[:：]\s*(.+?)(?:\n#|\n\n|\Z)', instruction, re.DOTALL)
        if opening_match:
            expected_opening = opening_match.group(1).strip()
            agent_turns = [t for t in dialogue if t.role == "agent"]
            if agent_turns:
                checks.append(self.det_checker.check_opening_line(agent_turns[0], expected_opening))

        return checks

    def _compute_scores(self, all_checks: list[CheckItem], result: EvaluationResult):
        """机械算分 - 从判定结果中确定性计算"""
        # 按维度汇总
        dim_scores = {}
        for c in all_checks:
            if c.dimension not in dim_scores:
                dim_scores[c.dimension] = {"score": 0.0, "max": 0.0}
            weighted = c.score * c.weight
            dim_scores[c.dimension]["score"] += weighted
            dim_scores[c.dimension]["max"] += c.weight  # max per item = 1.0 * weight

        result.dimension_scores = {}
        total_score = 0.0
        total_max = 0.0
        for dim, vals in dim_scores.items():
            rate = vals["score"] / vals["max"] if vals["max"] > 0 else 0.0
            result.dimension_scores[dim] = {
                "score": round(vals["score"], 2),
                "max": round(vals["max"], 2),
                "rate": round(rate, 4),
            }
            total_score += vals["score"]
            total_max += vals["max"]

        result.total_score = round(total_score, 2)
        result.max_score = round(total_max, 2)
        result.score_rate = round(total_score / total_max, 4) if total_max > 0 else 0.0

    def _generate_report(self, result: EvaluationResult, instruction: str) -> str:
        """生成可解释评测报告"""
        lines = []
        lines.append("=" * 60)
        lines.append(f"外呼对话模型指令遵循评测报告")
        lines.append(f"指令ID: {result.instruction_id}")
        lines.append("=" * 60)
        lines.append("")

        # 总分
        lines.append(f"【总分】{result.total_score:.1f} / {result.max_score:.1f}（{result.score_rate:.1%}）")
        lines.append("")

        # 各维度
        lines.append("【各维度得分】")
        for dim, vals in result.dimension_scores.items():
            bar = "█" * int(vals["rate"] * 20) + "░" * (20 - int(vals["rate"] * 20))
            lines.append(f"  {dim}: {vals['score']:.1f}/{vals['max']:.1f} {bar} {vals['rate']:.1%}")
        lines.append("")

        # Agent 判定明细
        lines.append("【Agent 逐项判定明细】")
        for c in result.checks:
            icon = {"pass": "✅", "partial": "⚠️", "fail": "❌"}.get(c.verdict, "❓")
            lines.append(f"  {icon} [{c.dimension}] {c.description}")
            lines.append(f"     证据: {c.evidence}")
            lines.append(f"     理由: {c.reason}")
            lines.append("")

        # 确定性检查明细
        if result.deterministic_checks:
            lines.append("【确定性约束检查明细】")
            for c in result.deterministic_checks:
                icon = {"pass": "✅", "partial": "⚠️", "fail": "❌"}.get(c.verdict, "❓")
                lines.append(f"  {icon} [{c.dimension}] {c.description}")
                lines.append(f"     证据: {c.evidence}")
                lines.append(f"     理由: {c.reason}")
                lines.append("")

        return "\n".join(lines)

    # ---- 辅助方法 ----

    def _format_dialogue(self, dialogue: list[DialogueTurn]) -> str:
        lines = []
        for t in dialogue:
            role_label = "Agent(被测)" if t.role == "agent" else "User(用户)"
            lines.append(f"[第{t.turn_id}轮] {role_label}: {t.content}")
        return "\n".join(lines)

    def _parse_json_list(self, text: str) -> list:
        """从LLM回复中提取JSON数组"""
        # 尝试提取 ```json ... ``` 块
        m = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if m:
            text = m.group(1)
        # 尝试直接解析
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            # 尝试找到 [ ... ]
            m = re.search(r'\[[\s\S]*\]', text)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    pass
        return []

    def _parse_json_single(self, text: str) -> dict:
        """从LLM回复中提取单个JSON对象"""
        m = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if m:
            text = m.group(1)
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            m = re.search(r'\{[\s\S]*\}', text)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    pass
        return {}
