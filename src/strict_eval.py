"""
严格评测系统 v2 - 带统计严谨性

核心原则：
1. LLM评分不是黄金标准，需要多次评测取统计量
2. 报告均值+置信区间，不报单次分数
3. 评测者间一致性（ICC）作为评测可靠性指标
4. 确定性检查与LLM判定分离，确定性检查作为锚点
5. LLM偏差显式量化
"""

import sys
import os
import json
import time
import socket
import re
import statistics
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))


# ============================================================
# LLM 调用
# ============================================================

def resolve_ips(host="mmdcadamsminiserverproxy.polaris", port=25340):
    addrs = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)
    return list(set(a[4][0] for a in addrs))

IPS = resolve_ips()
EXTRA_HEADERS = {
    "Adams-Platform-User": os.environ.get("ADAMS_PLATFORM_USER", ""),
    "Adams-User-Token": os.environ.get("ADAMS_USER_TOKEN", ""),
    "Adams-Business": os.environ.get("ADAMS_BUSINESS", ""),
}

def call_llm(messages, max_tokens=8192, temperature=0.3):
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

    for ip in IPS:
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
            content = data["choices"][0]["message"].get("content") or ""
            return content
        except Exception:
            continue
        finally:
            s.close()
    raise RuntimeError("All IPs failed")


# ============================================================
# 确定性检查器（代码验证，不依赖LLM）
# ============================================================

def deterministic_checks(instruction: str, dialogue: list[dict]) -> list[dict]:
    """确定性约束检查 - 代码直接算，零LLM依赖"""
    results = []
    agent_turns = [(i, d) for i, d in enumerate(dialogue) if d["role"] == "agent"]

    # 1. 字数检查
    length_match = re.search(r'约\s*(\d+)\s*个?字', instruction)
    length_match2 = re.search(r'最多\s*(\d+)[-~～]\s*(\d+)\s*个?字', instruction)
    max_chars = None
    if length_match2:
        max_chars = int(length_match2.group(2))
    elif length_match:
        max_chars = int(length_match.group(1))

    if max_chars:
        violations = 0
        details = []
        for idx, turn in agent_turns:
            char_count = len(turn["content"])
            ok = char_count <= max_chars
            if not ok:
                violations += 1
            details.append({"turn": turn.get("turn", idx+1), "chars": char_count, "limit": max_chars, "pass": ok})
        pass_count = len(details) - violations
        total = len(details)
        rate = pass_count / total if total > 0 else 1.0
        results.append({
            "dimension": "约束遵循度",
            "item": f"字数限制(≤{max_chars}字)",
            "type": "deterministic",
            "pass_rate": rate,
            "pass_count": pass_count,
            "total": total,
            "violations": violations,
            "details": details,
        })

    # 2. 禁用词检查
    banned = []
    for pattern in [r'不说[""\u201c\u201d]([^""\u201c\u201d]+)[""\u201c\u201d]']:
        banned.extend(re.findall(pattern, instruction))
    # 常见语气词禁用
    if '哈哈' in instruction or '语气词' in instruction:
        for w in ['好的', '哈哈', '嘿嘿', '嘻嘻']:
            if w in instruction and w not in banned:
                banned.append(w)

    if banned:
        violations = 0
        details = []
        for idx, turn in agent_turns:
            found = [w for w in banned if w in turn["content"]]
            ok = len(found) == 0
            if not ok:
                violations += 1
            details.append({"turn": turn.get("turn", idx+1), "found": found, "pass": ok})
        pass_count = len(details) - violations
        total = len(details)
        rate = pass_count / total if total > 0 else 1.0
        results.append({
            "dimension": "约束遵循度",
            "item": f"禁用词{banned}",
            "type": "deterministic",
            "pass_rate": rate,
            "pass_count": pass_count,
            "total": total,
            "violations": violations,
            "details": details,
        })

    # 3. 开场白关键信息覆盖
    opening_match = re.search(r'#?\s*Opening Line[:：]\s*(.+?)(?:\n#|\n\n|\Z)', instruction, re.DOTALL)
    if opening_match:
        expected = opening_match.group(1).strip()
        # 提取关键片段（跳过变量）
        expected_clean = re.sub(r'\$\{[^}]+\}', '', expected)
        expected_clean = re.sub(r'\*\*[^*]+\*\*', '', expected_clean)
        keywords = [w for w in re.split(r'[，。？！、\s]+', expected_clean) if len(w) >= 2]

        if agent_turns and keywords:
            actual = agent_turns[0][1]["content"]
            matched = [k for k in keywords if k in actual]
            rate = len(matched) / len(keywords)
            results.append({
                "dimension": "开场白准确度",
                "item": "开场白关键信息覆盖",
                "type": "deterministic",
                "pass_rate": rate,
                "matched": matched,
                "missing": [k for k in keywords if k not in matched],
                "keywords_total": len(keywords),
            })

    return results


# ============================================================
# LLM 评测 Prompt
# ============================================================

EVAL_PROMPT = """你是一个对话质量评测专家。请根据任务指令和对话记录，严格逐项评估。

【任务指令】
{instruction}

【对话记录】
{dialogue}

请从以下4个维度评估：

1. **流程遵循度**(权重2.0): Agent是否按指令中的Call Flow步骤执行？每步是否覆盖？条件分支是否正确？关键：步骤之间是否有节奏，是否给用户发言机会？
2. **信息传达完整性**(权重2.0): 指令Knowledge Points中的关键信息是否传达？参数是否准确？
   ⚠️ 重要：信息不仅要"说了"，还要"说得对"：
   - 信息是否在正确的流程步骤中传达？（例如：合同详情应在Step1-2中逐步说明，而非开场白一次性倾倒）
   - 信息是否以正确方式传达？（分步传达vs一次性倾倒——后者虽然在单轮中信息更全，但违反了流程约束，应扣分）
   - 如果某条信息虽然在对话中出现了，但违反了流程步骤的要求（如本应分步却一次说完），应判partial而非pass
3. **约束遵循度**(权重1.5): 语气、重复避免、超范围处理、特殊情况处理等语义约束是否遵守？（字数/禁用词由代码单独检查，此处评估语义层面的约束）
4. **任务达成度**(权重1.5): 核心任务是否完成？对话是否正常结束？用户是否理解了关键信息？

对每个维度列出具体检查项，每项判定pass/partial/fail，附原文证据。

输出JSON：
{{
  "dimensions": {{
    "流程遵循度": {{
      "items": [{{"name": "...", "verdict": "pass/partial/fail", "evidence": "原文引用", "reason": "理由"}}],
      "score": 0.0-1.0
    }},
    ...
  }},
  "overall_score": 0.0-1.0
}}

评分标准：score = (pass数 + 0.5*partial数) / 总项数
overall_score = 各维度score的加权和 / 总权重

只输出JSON。"""


# ============================================================
# 统计工具
# ============================================================

def mean(vals):
    return sum(vals) / len(vals) if vals else 0

def std(vals):
    if len(vals) < 2:
        return 0
    m = mean(vals)
    return (sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5

def confidence_interval_95(vals):
    """95% 置信区间 (t分布近似，样本小时偏保守)"""
    if len(vals) < 2:
        return mean(vals), mean(vals)
    m = mean(vals)
    s = std(vals)
    n = len(vals)
    # t值近似: n=3 -> 4.303, n=5 -> 2.776, n=10 -> 2.262
    t_table = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
               8: 2.365, 9: 2.306, 10: 2.262, 15: 2.145, 20: 2.093, 30: 2.042}
    t = t_table.get(n, 2.042) if n <= 30 else 1.96
    ci = t * s / (n ** 0.5)
    return max(0, m - ci), min(1, m + ci)

def icc_1way(vals_list):
    """
    单因素随机效应ICC(1) - 衡量多次评测间的一致性
    vals_list: [[run1_dim1, run1_dim2, ...], [run2_dim1, run2_dim2, ...], ...]
    返回: ICC值 (0-1, >0.75为好)
    """
    if len(vals_list) < 2:
        return None
    n_raters = len(vals_list)
    n_items = len(vals_list[0])
    if n_items < 2:
        return None

    grand_mean = mean([v for run in vals_list for v in run])

    # Between-items variance
    item_means = [mean([vals_list[r][i] for r in range(n_raters)]) for i in range(n_items)]
    ms_between = n_raters * sum((im - grand_mean) ** 2 for im in item_means) / (n_items - 1)

    # Within-items (residual) variance
    ms_within = sum((vals_list[r][i] - item_means[i]) ** 2
                    for r in range(n_raters) for i in range(n_items)) / (n_items * (n_raters - 1))

    if ms_between + ms_within == 0:
        return 1.0

    return (ms_between - ms_within) / (ms_between + (n_raters - 1) * ms_within)


# ============================================================
# 严格评测主流程
# ============================================================

def parse_eval_result(text):
    m = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if m:
        text = m.group(1)
    m = re.search(r'\{[\s\S]*"overall_score"[\s\S]*\}', text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None

def format_dialogue(dialogue):
    lines = []
    for d in dialogue:
        role = "Agent(被测)" if d["role"] == "agent" else "User(用户)"
        lines.append(f"[第{d.get('turn', 0)}轮] {role}: {d['content']}")
    return "\n".join(lines)

def run_single_llm_eval(instruction, dialogue_text, run_id=0):
    """单次LLM评测"""
    prompt = EVAL_PROMPT.format(instruction=instruction, dialogue=dialogue_text)
    content = call_llm([{"role": "user", "content": prompt}], temperature=0.3, max_tokens=8192)
    parsed = parse_eval_result(content)
    if parsed:
        dims = parsed.get("dimensions", {})
        return {
            "overall": parsed.get("overall_score"),
            "dimensions": {k: v.get("score") for k, v in dims.items()},
            "parsed": parsed,
        }
    return None

def strict_evaluate(instruction, dialogue, n_runs=5, label=""):
    """
    严格评测：多次运行 + 统计分析

    输出：
    - 各维度均值 + 95%置信区间
    - 总分均值 + 95%置信区间
    - ICC (评测者间一致性)
    - 确定性检查结果（锚点）
    - LLM偏差分析
    """
    dialogue_text = format_dialogue(dialogue)

    print(f"\n{'='*60}")
    print(f"严格评测: {label}")
    print(f"运行次数: {n_runs}")
    print(f"{'='*60}")

    # 1. 确定性检查（锚点，不依赖LLM）
    det_results = deterministic_checks(instruction, dialogue)
    print(f"\n--- 确定性检查（代码验证，零LLM偏差）---")
    for d in det_results:
        status = "✅" if d["pass_rate"] >= 0.8 else "⚠️" if d["pass_rate"] >= 0.5 else "❌"
        print(f"  {status} {d['item']}: {d['pass_rate']:.1%}")
        if "details" in d:
            for dd in d["details"]:
                if not dd.get("pass", True):
                    print(f"     违规: 第{dd.get('turn','')}轮")

    # 2. 多次LLM评测
    print(f"\n--- LLM评测 (n={n_runs}) ---")
    llm_results = []
    for i in range(n_runs):
        print(f"  运行 {i+1}/{n_runs}...", end=" ", flush=True)
        t0 = time.time()
        r = run_single_llm_eval(instruction, dialogue_text, run_id=i)
        elapsed = time.time() - t0
        if r:
            llm_results.append(r)
            print(f"总分={r['overall']:.3f} ({elapsed:.1f}s)")
        else:
            print(f"解析失败 ({elapsed:.1f}s)")

    if len(llm_results) < 2:
        print("LLM评测成功次数不足2次，无法计算统计量")
        return None

    # 3. 统计分析
    dim_names = ["流程遵循度", "信息传达完整性", "约束遵循度", "任务达成度"]

    # 收集各维度和总分的多次评测值
    overall_vals = [r["overall"] for r in llm_results if r["overall"] is not None]
    dim_vals = {}
    for dim in dim_names:
        dim_vals[dim] = [r["dimensions"].get(dim) for r in llm_results if r["dimensions"].get(dim) is not None]

    # 3a. 确定性检查校准：LLM约束遵循度不能超过确定性检查上界
    det_constraint_rate = None
    for d in det_results:
        if d["dimension"] == "约束遵循度":
            det_constraint_rate = d["pass_rate"]
            break

    if det_constraint_rate is not None and "约束遵循度" in dim_vals:
        calibrated_vals = []
        adjusted_count = 0
        for v in dim_vals["约束遵循度"]:
            if v > det_constraint_rate:
                adjusted_count += 1
                calibrated_vals.append(det_constraint_rate)
            else:
                calibrated_vals.append(v)
        if adjusted_count > 0:
            print(f"  [校准] 约束遵循度LLM评分被确定性检查上界约束: {adjusted_count}/{len(dim_vals['约束遵循度'])}次被下调至{det_constraint_rate:.1%}")
        dim_vals["约束遵循度"] = calibrated_vals
        # 更新 llm_results 中的维度分数
        for r in llm_results:
            if r["dimensions"].get("约束遵循度") is not None and r["dimensions"]["约束遵循度"] > det_constraint_rate:
                r["dimensions"]["约束遵循度"] = det_constraint_rate

    # 3b. 跨维度惩罚：如果信息"分步传达"被判fail，信息传达完整性打折
    # 原理：信息虽然在对话中出现了，但如果是违规的一次性倾倒，不应得高分
    if "信息传达完整性" in dim_vals and "流程遵循度" in dim_vals:
        info_penalized = []
        penalized_count = 0
        for i, r in enumerate(llm_results):
            # 检查是否有"信息分步传达"被判定fail
            parsed = r.get("parsed", {})
            has_delivery_fail = False
            for dim_data in parsed.get("dimensions", {}).values():
                for item in dim_data.get("items", []):
                    name = item.get("name", "")
                    if "分步" in name and item.get("verdict") == "fail":
                        has_delivery_fail = True
                        break
                if has_delivery_fail:
                    break

            info_score = dim_vals["信息传达完整性"][i] if i < len(dim_vals["信息传达完整性"]) else None
            flow_score = dim_vals["流程遵循度"][i] if i < len(dim_vals["流程遵循度"]) else None

            if has_delivery_fail and info_score is not None and flow_score is not None:
                # 信息传达完整性不能超过流程遵循度（因为信息是违规方式传达的）
                penalized = min(info_score, flow_score * 0.8)
                if penalized < info_score:
                    penalized_count += 1
                info_penalized.append(penalized)
            elif info_score is not None:
                info_penalized.append(info_score)

        if penalized_count > 0:
            print(f"  [跨维度惩罚] 信息传达完整性因分步传达违规被下调: {penalized_count}/{len(info_penalized)}次")
        dim_vals["信息传达完整性"] = info_penalized
        for i, r in enumerate(llm_results):
            if i < len(info_penalized):
                r["dimensions"]["信息传达完整性"] = info_penalized[i]

    # 重新计算总分（因为约束遵循度可能被校准了）
    weights = {"流程遵循度": 2.0, "信息传达完整性": 2.0, "约束遵循度": 1.5, "任务达成度": 1.5}
    recalculated_overall = []
    for r in llm_results:
        weighted_sum = sum(r["dimensions"].get(d, 0) * w for d, w in weights.items() if r["dimensions"].get(d) is not None)
        total_weight = sum(w for d, w in weights.items() if r["dimensions"].get(d) is not None)
        if total_weight > 0:
            recalculated_overall.append(weighted_sum / total_weight)
    if recalculated_overall:
        overall_vals = recalculated_overall

    # 均值 + 置信区间
    print(f"\n--- 统计结果 (n={len(llm_results)}) ---")
    print(f"{'指标':<16} {'均值':>6} {'标准差':>6} {'95%CI':>16}")
    print("-" * 50)

    results_summary = {}

    # 总分
    if overall_vals:
        m = mean(overall_vals)
        s = std(overall_vals)
        ci_lo, ci_hi = confidence_interval_95(overall_vals)
        print(f"{'总分(overall)':<16} {m:>6.3f} {s:>6.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")
        results_summary["overall"] = {"mean": m, "std": s, "ci_95": [ci_lo, ci_hi]}

    # 各维度
    for dim in dim_names:
        vals = dim_vals.get(dim, [])
        if vals:
            m = mean(vals)
            s = std(vals)
            ci_lo, ci_hi = confidence_interval_95(vals)
            print(f"{dim:<16} {m:>6.3f} {s:>6.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")
            results_summary[dim] = {"mean": m, "std": s, "ci_95": [ci_lo, ci_hi]}

    # 4. ICC（评测者间一致性）
    # 构造矩阵: runs × dimensions
    if len(llm_results) >= 3:
        matrix = []
        for r in llm_results:
            row = [r["dimensions"].get(dim, 0) for dim in dim_names if r["dimensions"].get(dim) is not None]
            if len(row) == len(dim_names):
                matrix.append(row)
        if len(matrix) >= 3:
            icc_val = icc_1way(matrix)
            if icc_val is not None:
                reliability = "优秀(>0.75)" if icc_val > 0.75 else "良好(0.5-0.75)" if icc_val > 0.5 else "差(<0.5)"
                print(f"\n  评测者间一致性 ICC(1) = {icc_val:.3f} ({reliability})")
                results_summary["icc"] = icc_val

    # 5. LLM偏差分析：和确定性检查对比
    print(f"\n--- LLM偏差分析 ---")
    if det_results:
        # 确定性约束遵循度
        det_pass_rates = [d["pass_rate"] for d in det_results if d["dimension"] == "约束遵循度"]
        if det_pass_rates:
            det_avg = mean(det_pass_rates)
            llm_constraint = dim_vals.get("约束遵循度", [])
            if llm_constraint:
                llm_avg = mean(llm_constraint)
                bias = llm_avg - det_avg
                direction = "偏宽容" if bias > 0.05 else "偏严格" if bias < -0.05 else "基本一致"
                print(f"  约束遵循度: 确定性检查={det_avg:.1%}, LLM评测={llm_avg:.1%}, 偏差={bias:+.1%} ({direction})")
                results_summary["llm_bias"] = {"deterministic": det_avg, "llm": llm_avg, "bias": bias, "direction": direction}

    # 6. 逐项判定频率统计
    print(f"\n--- 逐项判定频率 ---")
    item_verdicts = {}  # item_name -> [verdict_list across runs]
    for r in llm_results:
        dims = r.get("parsed", {}).get("dimensions", {})
        for dim_name, dim_data in dims.items():
            for item in dim_data.get("items", []):
                name = item.get("name", "")
                verdict = item.get("verdict", "fail")
                if name not in item_verdicts:
                    item_verdicts[name] = []
                item_verdicts[name].append(verdict)

    stable_count = 0
    unstable_count = 0
    for name, verdicts in sorted(item_verdicts.items(), key=lambda x: -len(set(x[1]))):
        if len(verdicts) < 2:
            continue
        unique = set(verdicts)
        pass_rate = verdicts.count("pass") / len(verdicts)
        if len(unique) == 1:
            stable_count += 1
            icon = "📌"
        else:
            unstable_count += 1
            icon = "🔀"
        print(f"  {icon} {name}: {'/'.join(verdicts)} (pass率={pass_rate:.0%})")

    stability_rate = stable_count / (stable_count + unstable_count) if (stable_count + unstable_count) > 0 else 0
    print(f"\n  判定稳定性: {stable_count}/{stable_count+unstable_count} 项稳定 ({stability_rate:.0%})")
    results_summary["stability_rate"] = stability_rate

    return results_summary


# ============================================================
# Main
# ============================================================

INSTRUCTION = """# Role
你是美团外卖骑手的站长。

# Task
致电"飞毛腿"骑手，通知他们今天合同已成功签署，并提醒他们完成配送任务。

# Opening Line
你好，请问是${rider_name}吗？我是站长。我看到你已报名飞毛腿。请记住，午餐和晚餐高峰期需要上线。单日合同每天至少完成 **X 单**；多日合同每天至少完成 **Y 单**。

# Call Flow
1. 告知骑手今天飞毛腿合同已生效，并询问他们是否可以开始配送。
2. 说明单日飞毛腿合同需要**连续 Y 天**完成配送；否则合同将受到影响。
3. 尽量挽留不想配送的骑手，鼓励能配送的骑手，并提醒他们注意安全。
4. 说明飞毛腿报名是按排名进行的，并非站长干预。骑手应减少拒单、取消和超时。在恶劣天气下工作、订单量更高，有助于保住飞毛腿资格。

# Knowledge Points (FAQ)
- 目前，许多骑手正在申请飞毛腿。如果你无法连续配送 **Y 天**，你的名额可能会被他人占用。
- 单日合同：在生效当天必须完成 **X 单**，否则合同及派单可能受到影响。
- 多日合同：每天必须完成 **Y 单**，否则后续合同及派单可能受到影响。
- 如需退出飞毛腿，必须在前一天 **Z 点之前**在 App 的"飞毛腿报名"中取消；次日生效。
- 连续完成 **W 天**多日合同，且每天完成 **Y 单**，将获得额外奖励（例如，与单日合同相比每单多 +$ 元）。

# Constraints
- 遵循对话流程和常见问题解答。
- 如被问及超出职责范围的问题，回复："我向同事确认后再回电给你。我现在能回答的先回答。"
- 保持语气随意，像打电话一样自然。
- 每次回复控制在**约 30 个字以内**。
- 避免重复回复；如需重申，请换种方式礼貌表达。
- 如果骑手坚持确实无法配送，安慰他们后挂断电话。"""

DIALOGUE_GOOD = [
    {"role": "agent", "content": "你好，请问是张三吗？我是站长。你已报名飞毛腿，午餐晚餐高峰期要上线。单日每天至少完成5单，多日每天至少完成3单。", "turn": 1},
    {"role": "user", "content": "哦，好的，合同生效了吗？", "turn": 2},
    {"role": "agent", "content": "生效了！你今天能开始配送吗？", "turn": 3},
    {"role": "user", "content": "可以的，不过多日合同要连续几天？", "turn": 4},
    {"role": "agent", "content": "多日合同要连续7天完成配送，不然会影响合同。", "turn": 5},
    {"role": "user", "content": "那如果我想退出怎么办？", "turn": 6},
    {"role": "agent", "content": "前一天晚上8点前在App的飞毛腿报名里取消就行。", "turn": 7},
    {"role": "user", "content": "好的，那安全方面有什么注意的吗？", "turn": 8},
    {"role": "agent", "content": "注意安全！减少拒单和超时，恶劣天气单多，有助于保住资格。加油！", "turn": 9},
    {"role": "user", "content": "行，谢谢站长！", "turn": 10},
    {"role": "agent", "content": "不客气，出发吧！", "turn": 11},
]

DIALOGUE_BAD = [
    {"role": "agent", "content": "喂，张三吗？我是站长。你报了飞毛腿，单日每天至少完成5单，多日每天至少完成3单，午餐晚餐高峰期要上线知道吧？另外合同已经生效了，你今天能开始配送吗？多日合同需要连续7天完成配送否则会影响合同的，如果你无法连续配送7天你的名额可能会被他人占用的，目前很多骑手都在申请飞毛腿。如果你需要退出的话前一天晚上8点之前在App里取消就行，连续完成7天多日合同每天完成3单的话还有额外奖励每单多2元。好的，你还有其他问题吗？", "turn": 1},
    {"role": "user", "content": "太多了我记不住...", "turn": 2},
    {"role": "agent", "content": "好的好的，没关系我再说一遍，单日每天5单多日每天3单连续7天...", "turn": 3},
    {"role": "user", "content": "我有点事，能稍后再说吗？", "turn": 4},
    {"role": "agent", "content": "好的，那我就先不说了，嗯好的，哈哈，拜拜。", "turn": 5},
]


def main():
    print("严格评测系统 v2")
    print("原则: LLM评分不是黄金标准，需要统计验证\n")

    n_runs = 5

    # 评测好对话
    result_good = strict_evaluate(INSTRUCTION, DIALOGUE_GOOD, n_runs=n_runs, label="好对话")

    # 评测差对话
    result_bad = strict_evaluate(INSTRUCTION, DIALOGUE_BAD, n_runs=n_runs, label="差对话")

    # 关键对比
    if result_good and result_bad:
        print("\n" + "=" * 60)
        print("关键对比")
        print("=" * 60)

        g_overall = result_good.get("overall", {})
        b_overall = result_bad.get("overall", {})

        if g_overall and b_overall:
            g_mean = g_overall["mean"]
            b_mean = b_overall["mean"]
            gap = g_mean - b_mean

            # 置信区间是否重叠
            g_ci = g_overall["ci_95"]
            b_ci = b_overall["ci_95"]
            overlap = not (g_ci[0] > b_ci[1] or b_ci[0] > g_ci[1])

            print(f"\n  好对话总分: {g_mean:.3f} 95%CI[{g_ci[0]:.3f}, {g_ci[1]:.3f}]")
            print(f"  差对话总分: {b_mean:.3f} 95%CI[{b_ci[0]:.3f}, {b_ci[1]:.3f}]")
            print(f"  差距: {gap:+.3f}")
            print(f"  置信区间重叠: {'是 ⚠️ 区分不显著' if overlap else '否 ✅ 区分显著'}")

            # Effect size (Cohen's d)
            pooled_std = ((g_overall.get("std", 0) ** 2 + b_overall.get("std", 0) ** 2) / 2) ** 0.5
            cohens_d = gap / pooled_std if pooled_std > 0 else float('inf')
            effect = "大" if abs(cohens_d) > 0.8 else "中" if abs(cohens_d) > 0.5 else "小"
            print(f"  Cohen's d: {cohens_d:.2f} (效应量: {effect})")

        # 各维度对比
        dims = ["流程遵循度", "信息传达完整性", "约束遵循度", "任务达成度"]
        print(f"\n  {'维度':<16} {'好对话':>8} {'差对话':>8} {'差距':>8}")
        print("  " + "-" * 44)
        for dim in dims:
            g = result_good.get(dim, {}).get("mean", "N/A")
            b = result_bad.get(dim, {}).get("mean", "N/A")
            if isinstance(g, (int, float)) and isinstance(b, (int, float)):
                print(f"  {dim:<16} {g:>8.3f} {b:>8.3f} {g-b:>+8.3f}")

        # ICC
        g_icc = result_good.get("icc")
        b_icc = result_bad.get("icc")
        g_icc_str = f"{g_icc:.3f}" if g_icc is not None else "N/A"
        b_icc_str = f"{b_icc:.3f}" if b_icc is not None else "N/A"
        print(f"\n  评测者一致性 ICC: 好对话={g_icc_str}, 差对话={b_icc_str}")

        # LLM偏差
        g_bias = result_good.get("llm_bias", {})
        b_bias = result_bad.get("llm_bias", {})
        if g_bias or b_bias:
            print(f"\n  LLM偏差: 好对话={g_bias.get('direction','N/A')}, 差对话={b_bias.get('direction','N/A')}")

    # 保存
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "strict_eval_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"good": result_good, "bad": result_bad}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output_path}")


if __name__ == "__main__":
    main()
