"""
用户模拟器 v2 — 参数化Persona + Sim2Real Gap缓解 + 行为度量

核心升级（基于最佳论文方法）：
1. 参数化Persona生成: 从固定5种persona升级为连续维度参数化
   - assertiveness: 支配性 (0-1, 0=完全被动, 1=完全主导)
   - receptivity: 接受度 (0-1, 0=完全拒绝, 1=完全配合)
   - urgency: 紧迫性 (0-1, 0=悠闲, 1=赶时间)
   - sophistication: 理解力 (0-1, 0=完全不懂, 1=专家)
   灵感: Gromada et al. EMNLP 2025 — Persona-driven User Simulation

2. Sim2Real Gap缓解:
   - 抵抗模式: 用户不会总是配合，会有自然的抵触和犹豫
   - 信息保留: 用户不会总是记住所有信息，需要重复
   - 分心行为: 用户可能走神、岔开话题
   - 情绪波动: 用户可能因各种原因产生负面情绪
   灵感: Zhou et al. 2026 — Mind the Sim2Real Gap

3. 行为度量指标(USI风格):
   - words_per_turn: 每轮字数分布
   - pushback_rate: 反驳/质疑比例
   - clarification_rate: 追问比例
   - early_termination_rate: 提前结束比例
   灵感: Zhou et al. 2026 — User-Sim Index

4. Persona一致性验证: 模拟结束后检查模拟器是否保持在角色内
"""

import json
import os
import socket
import re
import time
import random

EXTRA_HEADERS = {
    "Adams-Platform-User": os.environ.get("ADAMS_PLATFORM_USER", ""),
    "Adams-User-Token": os.environ.get("ADAMS_USER_TOKEN", ""),
    "Adams-Business": os.environ.get("ADAMS_BUSINESS", ""),
}

IPS = None

def get_ips():
    global IPS
    if IPS is None:
        IPS = list(set(a[4][0] for a in socket.getaddrinfo("mmdcadamsminiserverproxy.polaris", 25340, socket.AF_INET, socket.SOCK_STREAM)))
    return IPS

def call_llm(messages, max_tokens=4096, temperature=0.7):
    ips = get_ips()
    payload = json.dumps({
        "model": "kimi_k2d6",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"thinking": False},
    })
    body_path = "/service/20839/v1/chat/completions"
    http_req = f"POST {body_path} HTTP/1.1\r\nHost: mmdcadamsminiserverproxy.polaris:25340\r\nContent-Type: application/json\r\nConnection: close\r\n"
    for k, v in EXTRA_HEADERS.items():
        http_req += f"{k}: {v}\r\n"
    http_req += f"Content-Length: {len(payload)}\r\n\r\n{payload}"

    for ip in ips:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(120)
        try:
            s.connect((ip, 25340))
            s.sendall(http_req.encode())
            resp = b""
            while True:
                try:
                    chunk = s.recv(65536)
                    if not chunk:
                        break
                    resp += chunk
                except socket.timeout:
                    break
            decoded = resp.decode("utf-8", errors="replace")
            parts = decoded.split("\r\n\r\n", 1)
            if len(parts) < 2:
                continue
            body = parts[1]
            if "transfer-encoding: chunked" in parts[0].lower():
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
            data = json.loads(body)
            return data["choices"][0]["message"].get("content") or ""
        except Exception:
            continue
        finally:
            s.close()
    raise RuntimeError("All IPs failed")


# ============================================================
# 参数化Persona系统
# ============================================================

class PersonaConfig:
    """
    参数化Persona配置
    
    维度说明:
    - assertiveness: 支配性 (0=完全被动/顺从, 1=完全主导/强势)
    - receptivity: 接受度 (0=完全拒绝/抵触, 1=完全配合/接受)
    - urgency: 紧迫性 (0=悠闲/不急, 1=赶时间/想快速结束)
    - sophistication: 理解力 (0=完全不懂/需要解释, 1=专家/自己都懂)
    
    灵感: Gromada et al. EMNLP 2025 — assertiveness + precision of needs
    """
    def __init__(self, name: str, assertiveness: float = 0.5, receptivity: float = 0.5,
                 urgency: float = 0.3, sophistication: float = 0.5,
                 special_traits: list = None):
        self.name = name
        self.assertiveness = max(0, min(1, assertiveness))
        self.receptivity = max(0, min(1, receptivity))
        self.urgency = max(0, min(1, urgency))
        self.sophistication = max(0, min(1, sophistication))
        self.special_traits = special_traits or []
    
    def to_dict(self):
        return {
            "name": self.name,
            "assertiveness": self.assertiveness,
            "receptivity": self.receptivity,
            "urgency": self.urgency,
            "sophistication": self.sophistication,
            "special_traits": self.special_traits,
        }
    
    def describe_personality(self) -> str:
        """生成自然语言描述"""
        parts = []
        
        # 接受度描述
        if self.receptivity < 0.3:
            parts.append("你对这个电话很抵触，不太想配合，会找各种理由推脱")
        elif self.receptivity < 0.6:
            parts.append("你对这个电话不太确定，有犹豫，但也不是完全拒绝")
        elif self.receptivity < 0.8:
            parts.append("你比较配合，但会自然地问几个问题确认细节")
        else:
            parts.append("你很配合，愿意接受对方的安排，但也不会完全不问")
        
        # 支配性描述
        if self.assertiveness > 0.7:
            parts.append("你很强势，会主导对话方向，打断对方，坚持自己的想法")
        elif self.assertiveness > 0.4:
            parts.append("你有自己的主见，偶尔会追问或反驳，但不会太强势")
        else:
            parts.append("你比较被动，大多时候顺着对方说，不主动发起话题")
        
        # 紧迫性描述
        if self.urgency > 0.7:
            parts.append("你现在赶时间，希望对方快速说完重点，不耐烦长篇大论")
        elif self.urgency > 0.4:
            parts.append("你时间还好，但如果对方说太久会有点不耐烦")
        
        # 理解力描述
        if self.sophistication < 0.3:
            parts.append("你对这些规则和条款完全不懂，需要对方用很简单的话解释")
        elif self.sophistication > 0.7:
            parts.append("你对这些规则很了解，可能会问到一些细节和特殊情况")
        
        # 特殊特征
        for trait in self.special_traits:
            parts.append(trait)
        
        return "；".join(parts)
    
    def describe_resistance(self) -> str:
        """生成抵抗模式描述 — Sim2Real Gap缓解"""
        parts = []
        
        # 低接受度 → 更强的抵抗
        if self.receptivity < 0.3:
            parts.append("你会主动找理由拒绝，比如'太忙了''不想做''没兴趣'")
            parts.append("如果对方一直劝你，你会更烦躁而不是被说服")
        elif self.receptivity < 0.6:
            parts.append("你会犹豫，说'我再想想''不太确定'，需要对方给出具体理由才会考虑")
        
        # 高紧迫性 → 不耐烦
        if self.urgency > 0.7:
            parts.append("如果对方说了超过两句你还没听懂重点，你会说'能说快点吗''我有事'")
        
        # 低理解力 → 信息保留
        if self.sophistication < 0.3:
            parts.append("你不会记住对方说的所有细节，可能后面会再问一遍")
        
        return "；".join(parts) if parts else ""


# 预设Persona库 — 覆盖关键场景
PRESET_PERSONAS = {
    "cooperative": PersonaConfig("配合型", assertiveness=0.3, receptivity=0.85, urgency=0.2, sophistication=0.5),
    "reluctant": PersonaConfig("犹豫型", assertiveness=0.4, receptivity=0.35, urgency=0.3, sophistication=0.5,
                               special_traits=["你对能否完成配送任务不太确定，担心自己做不到"]),
    "rejecting": PersonaConfig("拒绝型", assertiveness=0.7, receptivity=0.15, urgency=0.5, sophistication=0.4,
                               special_traits=["你现在很忙不想接电话", "对方说什么你都会找理由推脱"]),
    "curious": PersonaConfig("追问型", assertiveness=0.6, receptivity=0.7, urgency=0.2, sophistication=0.8,
                             special_traits=["你对合同细节很好奇，会反复追问退出机制、奖励细节等"]),
    "driving": PersonaConfig("开车型", assertiveness=0.2, receptivity=0.6, urgency=0.9, sophistication=0.5,
                             special_traits=["你正在开车，不方便说话，只想听重点"]),
    "distracted": PersonaConfig("分心型", assertiveness=0.2, receptivity=0.5, urgency=0.4, sophistication=0.3,
                                special_traits=["你注意力不集中，会岔开话题，可能忘记对方刚说的内容", "你会问一些不相关的问题"]),
    "angry": PersonaConfig("不满型", assertiveness=0.8, receptivity=0.1, urgency=0.6, sophistication=0.6,
                           special_traits=["你之前有过不好的体验，对平台有怨气", "你会抱怨派单不公平或超时扣款"]),
    "elderly": PersonaConfig("老年型", assertiveness=0.2, receptivity=0.7, urgency=0.2, sophistication=0.1,
                             special_traits=["你年纪大了，不太会用App，听不太懂专业术语", "需要对方说慢一点、简单一点"]),
}

def generate_random_persona() -> PersonaConfig:
    """生成随机Persona — 增加测试多样性"""
    return PersonaConfig(
        name=f"随机用户_{random.randint(1000,9999)}",
        assertiveness=random.betavariate(2, 5),    # 多数人不太强势
        receptivity=random.betavariate(3, 2),      # 多数人比较配合
        urgency=random.betavariate(2, 5),          # 多数人不急
        sophistication=random.betavariate(3, 3),   # 中等理解力
    )


# ============================================================
# 对话模拟
# ============================================================

def build_user_system_prompt(persona: PersonaConfig, agent_role: str) -> str:
    """构建用户模拟器的system prompt — 包含Sim2Real Gap缓解机制"""
    
    personality = persona.describe_personality()
    resistance = persona.describe_resistance()
    
    prompt = f"""你是一个模拟真实用户的角色。你正在接听一个外呼电话。

【来电方】{agent_role}

【你的人物设定】
{personality}
"""

    if resistance:
        prompt += f"""
【你的抵抗模式 — 重要！】
{resistance}
"""

    prompt += """
【行为规则 — 模拟真实用户】
1. 像真实用户一样自然回复，用口语化短句
2. 严格按照你的人物设定行事，不要为了方便对方而改变性格
3. 不要主动推进对话，等对方说
4. 每次回复控制在20字以内（真实用户说话都很短）
5. 不要用书面语，用日常口语
6. 不要总是配合 — 真实用户经常：
   - 不听完就打断
   - 走神或岔开话题
   - 忘记对方刚说的内容
   - 表达不耐烦或困惑
   - 给模糊回答如"嗯""哦""再说吧"
7. 如果对方说了很多信息一次倾倒，你很可能会记不住，会要求对方慢点说或重复
8. 如果对方态度不好或太啰嗦，你会更不配合
9. 如果对方说完了你想说的，可以主动结束（说"好的挂了""再见"等）

⚠️ 关键：不要过度配合！不要变成一个完美的倾听者。真实用户是不完美的。

⚠️ 严禁元评论：
- 不要分析自己的角色或行为（如"用户说..."、"我需要..."、"作为..."）
- 不要解释你为什么这么说
- 不要在回复中包含任何思考过程
- 只输出你作为用户的自然口语回复，不要有任何额外文字"""

    return prompt


def build_agent_system_prompt(instruction: str) -> str:
    """构建被测Agent的system prompt"""
    return f"""你是一个外呼客服Agent，需要严格按照以下指令执行通话。

{instruction}

请严格按照指令中的Role、Call Flow、Knowledge Points和Constraints执行。
注意字数限制和语气要求。
每次回复后等待用户回应。

只输出你的回复内容，不要加引号、不要加角色标记。"""


def sanitize_user_reply(text: str) -> tuple[str, list[str]]:
    """清理模拟用户输出中的元评论和非口语化残留。"""
    issues = []
    raw = (text or "").strip().strip('"').strip()
    meta_patterns = [
        "用户说", "作为用户", "我需要", "我会", "我的角色", "对方说",
        "真实用户", "思考", "分析", "回复应该", "我作为", "接下来",
    ]
    if any(p in raw for p in meta_patterns):
        issues.append("检测到元评论残留")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    candidate = lines[-1] if lines else raw
    for sep in ["：", ":"]:
        if sep in candidate and any(candidate.startswith(p) for p in ["用户", "我", "回复", "自然回复"]):
            candidate = candidate.split(sep, 1)[1].strip()
    candidate = re.sub(r'^(用户说|用户|我|回复|自然回复)[：:：\s]*', '', candidate).strip()
    candidate = re.sub(r'["“”]', '', candidate).strip()
    if any(p in candidate for p in meta_patterns):
        candidate = re.split(r'[。！？!?]\s*', candidate)[-1].strip() or candidate
    if len(candidate) > 50:
        issues.append("用户回复过长，已截断")
        candidate = candidate[:50].rstrip('，。、；; ') + '...'
    if not candidate:
        candidate = "嗯"
        issues.append("空回复兜底")
    return candidate, issues


def simulator_quality_checks(dialogue: list[dict], persona: PersonaConfig, behavior_metrics: dict,
                             consistency: dict, sanitation_issues: list[dict]) -> dict:
    """输出用户模拟器自检结果，避免把不真实样本混入评测。"""
    issues = []
    user_turns = [d for d in dialogue if d["role"] == "user"]
    meta_patterns = ["用户说", "作为用户", "我需要", "真实用户", "思考", "分析"]
    for d in user_turns:
        if any(p in d["content"] for p in meta_patterns):
            issues.append(f"第{d.get('turn')}轮疑似元评论: {d['content'][:30]}")
    if behavior_metrics.get("max_words_per_turn", 0) > 60:
        issues.append("用户单轮回复过长，真实性偏低")
    if persona.receptivity < 0.3 and behavior_metrics.get("pushback_rate", 0) == 0:
        issues.append("低接受度 persona 未表现出抵触")
    if persona.sophistication < 0.3 and behavior_metrics.get("clarification_rate", 0) == 0:
        issues.append("低理解力 persona 未出现追问/澄清")
    if not consistency.get("consistent", True):
        issues.extend(consistency.get("issues", []))
    issues.extend(i.get("issue", "") for i in sanitation_issues if i.get("issue"))
    unique_issues = list(dict.fromkeys(issues))
    score = max(0.0, 1.0 - 0.15 * len(unique_issues))
    return {
        "score": score,
        "passed": score >= 0.7,
        "issues": unique_issues,
        "sanitized_turns": len(sanitation_issues),
    }


def compute_behavior_metrics(dialogue: list[dict]) -> dict:
    """
    计算行为度量指标 — USI风格
    
    灵感: Zhou et al. 2026 — User-Sim Index behavioral dimensions
    """
    user_turns = [d for d in dialogue if d["role"] == "user"]
    agent_turns = [d for d in dialogue if d["role"] == "agent"]
    
    if not user_turns:
        return {}
    
    # D1: Communication Style
    words_per_turn = [len(d["content"]) for d in user_turns]
    
    # D3: Clarification/Pushback
    clarification_keywords = ["什么", "为什么", "怎么", "哪", "多少", "不懂", "没听清", "再说一遍", "什么意思"]
    pushback_keywords = ["不行", "不要", "不想", "算了", "没必要", "太麻烦", "我不要", "拒绝"]
    
    clarification_count = 0
    pushback_count = 0
    for d in user_turns:
        content = d["content"]
        if any(kw in content for kw in clarification_keywords):
            clarification_count += 1
        if any(kw in content for kw in pushback_keywords):
            pushback_count += 1
    
    # D4: Error Reaction (情绪相关)
    emotion_keywords = ["烦", "气", "讨厌", "受不了", "太过分", "不满意", "投诉"]
    emotion_count = 0
    for d in user_turns:
        if any(kw in d["content"] for kw in emotion_keywords):
            emotion_count += 1
    
    # Short responses (单字或极短回复)
    short_responses = sum(1 for d in user_turns if len(d["content"]) <= 5)
    
    # Early termination
    last_user = user_turns[-1]["content"] if user_turns else ""
    end_keywords = ["挂了", "再见", "拜拜", "稍后再说", "回头聊", "开车不方便", "不方便接", "先这样"]
    early_termination = any(kw in last_user for kw in end_keywords)
    
    return {
        "n_user_turns": len(user_turns),
        "n_agent_turns": len(agent_turns),
        "avg_words_per_turn": sum(words_per_turn) / len(words_per_turn) if words_per_turn else 0,
        "max_words_per_turn": max(words_per_turn) if words_per_turn else 0,
        "clarification_rate": clarification_count / len(user_turns) if user_turns else 0,
        "pushback_rate": pushback_count / len(user_turns) if user_turns else 0,
        "emotion_rate": emotion_count / len(user_turns) if user_turns else 0,
        "short_response_rate": short_responses / len(user_turns) if user_turns else 0,
        "early_termination": early_termination,
    }


def verify_persona_consistency(dialogue: list[dict], persona: PersonaConfig, agent_role: str) -> dict:
    """
    Persona一致性验证 — 检查模拟器是否保持在角色内
    
    灵感: Gromada et al. 2025 — "dual assessment — simulator adherence to persona"
    """
    user_turns = [d for d in dialogue if d["role"] == "user"]
    if not user_turns:
        return {"consistent": True, "issues": []}
    
    issues = []
    
    # 检查1：低接受度用户不应该太配合
    if persona.receptivity < 0.3:
        cooperative_phrases = ["好的", "没问题", "可以", "行", "知道了"]
        cooperative_count = sum(1 for d in user_turns if any(p in d["content"] for p in cooperative_phrases))
        total = len(user_turns)
        if cooperative_count / total > 0.5:
            issues.append(f"低接受度用户({persona.receptivity:.1f})配合度过高: {cooperative_count}/{total}轮包含配合用语")
    
    # 检查2：高紧迫性用户不应该长篇大论
    if persona.urgency > 0.7:
        long_turns = sum(1 for d in user_turns if len(d["content"]) > 20)
        if long_turns > len(user_turns) * 0.3:
            issues.append(f"高紧迫性用户({persona.urgency:.1f})回复过长: {long_turns}轮超过20字")
    
    # 检查3：高支配性用户应该主动提问或主导
    if persona.assertiveness > 0.6:
        proactive_keywords = ["我要", "你必须", "不行", "听我说", "我要求"]
        proactive_count = sum(1 for d in user_turns if any(kw in d["content"] for kw in proactive_keywords))
        if proactive_count == 0:
            issues.append(f"高支配性用户({persona.assertiveness:.1f})没有表现出主动/强势行为")
    
    # 检查4：低理解力用户不应该说专业术语
    if persona.sophistication < 0.3:
        jargon = ["合同", "条款", "资格", "排名", "生效"]
        jargon_count = sum(1 for d in user_turns if any(j in d["content"] for j in jargon))
        if jargon_count > len(user_turns) * 0.3:
            issues.append(f"低理解力用户({persona.sophistication:.1f})使用了过多专业术语")
    
    return {
        "consistent": len(issues) == 0,
        "issues": issues,
        "persona": persona.to_dict(),
    }


def simulate_dialogue(instruction: str, persona: PersonaConfig = None, persona_type: str = "cooperative",
                      max_turns: int = 12, verbose: bool = True) -> dict:
    """
    用LLM模拟用户和被测Agent的对话
    
    返回: {
        "dialogue": [...],
        "persona": persona.to_dict(),
        "behavior_metrics": {...},
        "persona_consistency": {...},
    }
    """
    if persona is None:
        persona = PRESET_PERSONAS.get(persona_type, PRESET_PERSONAS["cooperative"])
    
    # 从指令中提取角色
    role_match = re.search(r'#?\s*Role[:：\n]\s*(.+?)(?:\n#|\n\n|\n[A-Z]|\Z)', instruction, re.DOTALL)
    agent_role = role_match.group(1).strip() if role_match else "客服"
    
    # 从指令中提取开场白
    opening_match = re.search(r'#?\s*Opening Line[:：]\s*(.+?)(?:\n#|\n\n|\Z)', instruction, re.DOTALL)
    opening_line = opening_match.group(1).strip() if opening_match else "您好"
    opening_line = re.sub(r'\$\{[^}]+\}', '张三', opening_line)
    # 替换加粗标记中的变量
    opening_line = re.sub(r'\*\*([^*]+)\*\*', lambda m: m.group(1), opening_line)
    
    # 构建system prompts
    user_system = build_user_system_prompt(persona, agent_role)
    agent_system = build_agent_system_prompt(instruction)
    
    dialogue = []
    
    # 第一轮：Agent发起开场白
    dialogue.append({"role": "agent", "content": opening_line, "turn": 1})
    
    # 交替对话
    user_messages = [{"role": "system", "content": user_system}]
    agent_messages = [{"role": "system", "content": agent_system}]
    sanitation_issues = []
    
    for turn in range(2, max_turns + 1):
        last_msg = dialogue[-1]["content"]
        last_role = dialogue[-1]["role"]
        
        if last_role == "agent":
            # 用户回复
            user_messages.append({"role": "user", "content": f"对方说: {last_msg}\n\n你的回复:"})
            try:
                raw_reply = call_llm(user_messages, max_tokens=200, temperature=0.7)
                user_reply, issues = sanitize_user_reply(raw_reply)
                for issue in issues:
                    sanitation_issues.append({"turn": turn, "issue": issue, "raw": raw_reply[:120]})
            except Exception:
                user_reply = "嗯"
                sanitation_issues.append({"turn": turn, "issue": "LLM调用失败兜底", "raw": ""})
            user_messages.append({"role": "assistant", "content": user_reply})
            dialogue.append({"role": "user", "content": user_reply, "turn": turn})
        else:
            # Agent回复
            agent_messages.append({"role": "user", "content": f"对方说: {dialogue[-1]['content']}\n\n你的回复:"})
            try:
                agent_reply = call_llm(agent_messages, max_tokens=200, temperature=0.5)
                agent_reply = agent_reply.strip().strip('"').strip()
            except Exception:
                agent_reply = "好的"
            agent_messages.append({"role": "assistant", "content": agent_reply})
            dialogue.append({"role": "agent", "content": agent_reply, "turn": turn})
        
        # 判断对话是否应该结束
        last_content = dialogue[-1]["content"]
        end_keywords = ["挂了", "再见", "拜拜", "稍后再说", "回头聊", "开车不方便", "不方便接"]
        if any(kw in last_content for kw in end_keywords) and dialogue[-1]["role"] == "user":
            break
        # 如果用户连续2次说"嗯""哦"等敷衍词，可能想结束
        if len(dialogue) >= 3:
            last_2_user = [d["content"] for d in dialogue[-2:] if d["role"] == "user"]
            passive_words = ["嗯", "哦", "好", "行"]
            if len(last_2_user) >= 2 and all(w in passive_words for w in last_2_user):
                if persona.urgency > 0.5 or persona.receptivity < 0.3:
                    break
        if turn >= max_turns:
            break
    
    # 计算行为度量
    behavior_metrics = compute_behavior_metrics(dialogue)
    
    # Persona一致性验证
    consistency = verify_persona_consistency(dialogue, persona, agent_role)
    quality = simulator_quality_checks(dialogue, persona, behavior_metrics, consistency, sanitation_issues)
    
    result = {
        "dialogue": dialogue,
        "persona": persona.to_dict(),
        "behavior_metrics": behavior_metrics,
        "persona_consistency": consistency,
        "simulator_quality": quality,
    }
    
    if verbose:
        print(f"\n  Persona: {persona.name} (assert={persona.assertiveness:.1f}, recept={persona.receptivity:.1f}, "
              f"urgency={persona.urgency:.1f}, soph={persona.sophistication:.1f})")
        print(f"  对话轮数: {len(dialogue)}")
        print(f"  行为指标: 平均{behavior_metrics.get('avg_words_per_turn', 0):.1f}字/轮, "
              f"追问率={behavior_metrics.get('clarification_rate', 0):.0%}, "
              f"反驳率={behavior_metrics.get('pushback_rate', 0):.0%}")
        if not consistency["consistent"]:
            print(f"  ⚠️ Persona一致性问题: {consistency['issues']}")
        if not quality["passed"]:
            print(f"  ⚠️ 模拟器质量问题: {quality['issues']}")
    
    return result


def generate_diverse_dialogues(instruction: str, n_dialogues: int = 8,
                                include_presets: bool = True) -> list[dict]:
    """
    生成多样化对话集
    
    策略: 预设Persona覆盖关键场景 + 随机Persona增加多样性
    
    灵感: Gromada et al. 2025 — persona diversity for evaluation coverage
    """
    results = []
    
    if include_presets:
        preset_keys = list(PRESET_PERSONAS.keys())[:n_dialogues]
        print(f"使用 {len(preset_keys)} 个预设Persona + 随机生成 {max(0, n_dialogues - len(preset_keys))} 个")
        
        for key in preset_keys:
            print(f"\n生成对话: {key}...", flush=True)
            try:
                result = simulate_dialogue(instruction, persona_type=key)
                result["persona_type"] = key
                results.append(result)
            except Exception as e:
                print(f"  生成失败: {e}")
    
    # 随机Persona补充
    remaining = n_dialogues - len(results)
    for i in range(remaining):
        persona = generate_random_persona()
        print(f"\n生成对话: 随机Persona #{i+1}...", flush=True)
        try:
            result = simulate_dialogue(instruction, persona=persona)
            result["persona_type"] = f"random_{i+1}"
            results.append(result)
        except Exception as e:
            print(f"  生成失败: {e}")
    
    return results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    INSTRUCTION = """# Role
你是美团外卖骑手的站长。

# Task
致电"飞毛腿"骑手，通知他们今天合同已成功签署，并提醒他们完成配送任务。

# Opening Line
你好，请问是${rider_name}吗？我是站长。我看到你已报名飞毛腿。请记住，午餐和晚餐高峰期需要上线。单日合同每天至少完成 **5 单**；多日合同每天至少完成 **3 单**。

# Call Flow
1. 告知骑手今天飞毛腿合同已生效，并询问他们是否可以开始配送。
2. 说明单日飞毛腿合同需要**连续 7 天**完成配送；否则合同将受到影响。
3. 尽量挽留不想配送的骑手，鼓励能配送的骑手，并提醒他们注意安全。
4. 说明飞毛腿报名是按排名进行的，并非站长干预。骑手应减少拒单、取消和超时。在恶劣天气下工作、订单量更高，有助于保住飞毛腿资格。

# Knowledge Points (FAQ)
- 目前，许多骑手正在申请飞毛腿。如果你无法连续配送 **7 天**，你的名额可能会被他人占用。
- 单日合同：在生效当天必须完成 **5 单**，否则合同及派单可能受到影响。
- 多日合同：每天必须完成 **3 单**，否则后续合同及派单可能受到影响。
- 如需退出飞毛腿，必须在前一天 **20 点之前**在 App 的"飞毛腿报名"中取消；次日生效。
- 连续完成 **7 天**多日合同，且每天完成 **3 单**，将获得额外奖励（例如，与单日合同相比每单多 **2 元**）。

# Constraints
- 遵循对话流程和常见问题解答。
- 如被问及超出职责范围的问题，回复："我向同事确认后再回电给你。我现在能回答的先回答。"
- 保持语气随意，像打电话一样自然。
- 每次回复控制在**约 30 个字以内**。
- 避免重复回复；如需重申，请换种方式礼貌表达。
- 如果骑手坚持确实无法配送，安慰他们后挂断电话。"""

    print("用户模拟器 v2 — 参数化Persona + Sim2Real Gap缓解")
    print("="*60)
    
    # 生成多样化对话
    dialogues = generate_diverse_dialogues(INSTRUCTION, n_dialogues=8, include_presets=True)
    
    # 汇总
    print(f"\n{'='*60}")
    print(f"生成完成: {len(dialogues)} 条对话")
    print(f"{'='*60}")
    
    # 行为度量汇总
    all_metrics = [d["behavior_metrics"] for d in dialogues if d.get("behavior_metrics")]
    if all_metrics:
        print(f"\n行为度量汇总:")
        print(f"  平均字数/轮: {sum(m.get('avg_words_per_turn',0) for m in all_metrics)/len(all_metrics):.1f}")
        print(f"  追问率: {sum(m.get('clarification_rate',0) for m in all_metrics)/len(all_metrics):.0%}")
        print(f"  反驳率: {sum(m.get('pushback_rate',0) for m in all_metrics)/len(all_metrics):.0%}")
        print(f"  提前结束率: {sum(1 for m in all_metrics if m.get('early_termination'))/len(all_metrics):.0%}")
    
    # Persona一致性汇总
    consistency_results = [d["persona_consistency"] for d in dialogues if d.get("persona_consistency")]
    if consistency_results:
        consistent_count = sum(1 for c in consistency_results if c.get("consistent"))
        print(f"  Persona一致性: {consistent_count}/{len(consistency_results)} 通过")
        issues = [issue for c in consistency_results for issue in c.get("issues", [])]
        if issues:
            print(f"  一致性问题:")
            for issue in issues:
                print(f"    - {issue}")
    
    # 保存
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    save_data = {
        "instruction": INSTRUCTION,
        "dialogues": [{
            "persona": d["persona"],
            "persona_type": d.get("persona_type", "unknown"),
            "dialogue": d["dialogue"],
                "behavior_metrics": d.get("behavior_metrics", {}),
                "persona_consistency": d.get("persona_consistency", {}),
                "simulator_quality": d.get("simulator_quality", {}),
            } for d in dialogues],

    }
    
    output_path = os.path.join(output_dir, "simulated_dialogues_v2.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output_path}")
