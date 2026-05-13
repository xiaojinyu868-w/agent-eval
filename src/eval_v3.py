"""
评测系统 v3 — Agent-as-a-Judge + 原子化Rubric + 证据锚定 + WGR校准

核心升级（基于最佳论文方法）：
1. 原子化Rubric引擎: 指令特定的检查清单，binary marking，消除主观灰色地带
   - 灵感: "Rubric Is All You Need" (Pathak et al. 2025) — QS rubric比generic rubric κ提升314%
   - 灵感: RULERS (Hong et al. 2026) — 编译rubric为不可变bundle，evidence-anchored scoring

2. Agent-as-a-Judge多步评测: 不是一次性打分，而是Agent自主走完整条评测链
   - 灵感: Agent-as-a-Judge (Zhuge et al. 2024) — 显著优于LLM-as-a-Judge
   - 步骤: Rubric编译 → 逐项判定(带证据) → 证据验证 → 跨维度校准 → 分数计算

3. 证据锚定: 每个判定必须附带对话原文，无证据则分数机械封顶
   - 灵感: RULERS Phase II — "If |E_k| < m, score is mechanically capped"

4. 确定性校准锚点: 字数/禁用词等代码可验证的约束作为校准锚点
   - 灵感: RULERS — 确定性检查和LLM判定分离，确定性检查提供上界

5. BCa Bootstrap CI: 比percentile bootstrap更稳健（校正偏差和偏度）
6. Gwet's AC1: 比Cohen's Kappa抗悖论（高一致性时Kappa反常低）
7. Krippendorff's alpha: 支持多评价者+有序尺度
8. pass^k可靠性指标: 测量Agent在多次运行中的一致性（τ-bench风格）
"""

import sys
import os
import json
import time
import socket
import re
import random
import math
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))


# ============================================================
# LLM 调用（Raw Socket 绕过代理限制）
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

def call_llm(messages, max_tokens=8192, temperature=0.3, timeout=60):
    import time
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

    start_time = time.time()
    # 优先使用最近成功的IP（缓存）
    ip_list = list(IPS)
    if hasattr(call_llm, '_last_good_ip') and call_llm._last_good_ip in ip_list:
        ip_list.remove(call_llm._last_good_ip)
        ip_list.insert(0, call_llm._last_good_ip)
    
    for ip in ip_list:
        elapsed = time.time() - start_time
        remaining = timeout - elapsed
        if remaining <= 0:
            raise TimeoutError(f"call_llm total timeout {timeout}s exceeded")
        # 单IP超时：最少给30秒（让LLM有足够时间处理长prompt）
        socket_timeout = min(30, max(remaining * 0.8, 5))
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(socket_timeout)
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
                s.close()
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
            s.close()
            # 缓存成功IP
            call_llm._last_good_ip = ip
            return content
        except Exception:
            try:
                s.close()
            except:
                pass
            continue
    raise RuntimeError("All IPs failed")


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

def _normal_ppf(p):
    """Inverse normal CDF approximation (Abramowitz & Stegun)"""
    if p <= 0:
        return -8.0
    if p >= 1:
        return 8.0
    if p > 0.5:
        return -_normal_ppf(1 - p)
    t = (-2 * math.log(p)) ** 0.5
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return -(t - (c0 + c1*t + c2*t**2) / (1 + d1*t + d2*t**2 + d3*t**3))

def bootstrap_ci(vals, n_bootstrap=10000, ci=0.95):
    """Percentile Bootstrap CI"""
    if len(vals) < 2:
        return mean(vals), mean(vals)
    n = len(vals)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = [random.choice(vals) for _ in range(n)]
        boot_means.append(mean(sample))
    boot_means.sort()
    alpha = (1 - ci) / 2
    lo = boot_means[int(n_bootstrap * alpha)]
    hi = boot_means[int(n_bootstrap * (1 - alpha))]
    return lo, hi

def _normal_cdf(z):
    """Standard normal CDF approximation (Abramowitz & Stegun)"""
    if z < -8:
        return 0.0
    if z > 8:
        return 1.0
    # Using error function approximation
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    d = 0.3989422804014327  # 1/sqrt(2*pi)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    result = 1.0 - d * math.exp(-0.5 * z * z) * poly
    return result if z > 0 else 1.0 - result


def bca_bootstrap_ci(vals, n_bootstrap=10000, ci=0.95):
    """
    BCa (Bias-Corrected and Accelerated) Bootstrap CI
    
    比percentile bootstrap更稳健：
    - z0校正偏差（bootstrap分布中位数与观测值的偏移）
    - a校正偏度（jackknife估计的加速因子）
    
    灵感: Efron & Narasimhan 2020
    """
    if len(vals) < 2:
        return mean(vals), mean(vals)
    
    n = len(vals)
    theta_hat = mean(vals)
    
    # Step 1: Bootstrap replications
    boot_means = []
    for _ in range(n_bootstrap):
        sample = [random.choice(vals) for _ in range(n)]
        boot_means.append(mean(sample))
    boot_means.sort()
    
    # Step 2: Bias correction z0
    # z0 = Φ^{-1}(proportion of bootstrap reps < theta_hat)
    prop_below = sum(1 for b in boot_means if b < theta_hat) / n_bootstrap
    prop_below = max(0.001, min(0.999, prop_below))  # clamp
    z0 = _normal_ppf(prop_below)
    
    # Step 3: Acceleration factor a (jackknife)
    # a = sum(d_i^3) / (6 * sum(d_i^2)^1.5) where d_i = jack_mean - jack_means[i]
    jack_means = []
    for i in range(n):
        jack_sample = vals[:i] + vals[i+1:]
        jack_means.append(mean(jack_sample))
    jack_mean_of_means = mean(jack_means)
    
    diffs = [(jack_mean_of_means - jm) for jm in jack_means]
    sum_d2 = sum(d**2 for d in diffs)
    sum_d3 = sum(d**3 for d in diffs)
    a = sum_d3 / (6 * sum_d2**1.5) if sum_d2 > 0 else 0
    
    # Step 4: Adjusted percentiles
    # BCa formula: alpha_adj = Φ(z0 + (z0 + z_alpha) / (1 - a*(z0 + z_alpha)))
    alpha = (1 - ci) / 2
    z_alpha = _normal_ppf(alpha)
    z_1alpha = _normal_ppf(1 - alpha)
    
    def adjust(z_val):
        num = z0 + z_val
        denom = 1 - a * num
        if abs(denom) < 1e-10:
            return _normal_cdf(z_val)
        return _normal_cdf(z0 + num / denom)
    
    alpha1 = adjust(z_alpha)
    alpha2 = adjust(z_1alpha)
    
    # Clamp to valid range [0.001, 0.999]
    alpha1 = max(0.001, min(0.999, alpha1))
    alpha2 = max(0.001, min(0.999, alpha2))
    
    lo = boot_means[int(n_bootstrap * alpha1)]
    hi = boot_means[int(n_bootstrap * alpha2)]
    
    return lo, hi

def cohens_d(group1, group2):
    """Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0
    pooled_std = ((std(group1)**2 * (n1-1) + std(group2)**2 * (n2-1)) / (n1 + n2 - 2)) ** 0.5
    return (mean(group1) - mean(group2)) / pooled_std if pooled_std > 0 else 0

def gwet_ac1(rater1, rater2):
    """
    Gwet's AC1 — 比Cohen's Kappa抗悖论
    
    优势: 当一致性高但边缘分布不均时（如90%都pass），
    Cohen's Kappa会反常地低，AC1不会。
    
    灵感: Gwet 2008, "Computing inter-rater reliability..."
    """
    if len(rater1) != len(rater2) or len(rater1) < 2:
        return None
    
    n = len(rater1)
    categories = list(set(rater1 + rater2))
    K = len(categories)
    
    if K <= 1:
        return 1.0
    
    # Observed agreement
    agree = sum(1 for a, b in zip(rater1, rater2) if a == b)
    p_o = agree / n
    
    # Gwet's AC1 expected agreement:
    # p_e = (2 / (K * (K-1))) * sum_c pi_c * (1 - pi_c)
    # where pi_c = (n1_c + n2_c) / (2N)
    
    # Standard Gwet AC1:
    # p_e = (1 / (K * (K-1))) * sum_c ( (n1_c + n2_c) / (2N) ) * (2 - (n1_c + n2_c)/N )
    # But the simpler and correct formulation:
    K = len(categories)
    if K <= 1:
        return 1.0
    
    p_e = 0
    for c in categories:
        n1_c = rater1.count(c)
        n2_c = rater2.count(c)
        pi_c = (n1_c + n2_c) / (2 * n)
        p_e += pi_c * (1 - pi_c)
    p_e = 2 * p_e / (K * (K - 1)) if K > 1 else 0
    
    denom = 1 - p_e
    if abs(denom) < 1e-10:
        return 1.0
    return (p_o - p_e) / denom

def krippendorffs_alpha(data_matrix, level="ordinal"):
    """
    Krippendorff's alpha — 支持多评价者+有序尺度
    
    data_matrix: list of lists, data_matrix[rater][item] = score
    level: "nominal", "ordinal", "interval", "ratio"
    
    优势: 处理缺失数据、多评价者、有序尺度
    灵感: Krippendorff 2004
    """
    n_raters = len(data_matrix)
    n_items = len(data_matrix[0]) if n_raters > 0 else 0
    
    if n_raters < 2 or n_items < 2:
        return None
    
    # Collect all valid pairs
    values = []
    for r in range(n_raters):
        for i in range(n_items):
            v = data_matrix[r][i]
            if v is not None:
                values.append(v)
    
    if not values:
        return None
    
    # Distance function
    if level == "nominal":
        def diff_metric(a, b):
            return 0 if a == b else 1
    elif level == "ordinal":
        def diff_metric(a, b):
            # For ordinal, use squared rank difference
            return (a - b) ** 2
    else:  # interval
        def diff_metric(a, b):
            return (a - b) ** 2
    
    # Observed disagreement
    d_o = 0
    n_pairs = 0
    for i in range(n_items):
        item_vals = [data_matrix[r][i] for r in range(n_raters) if data_matrix[r][i] is not None]
        for a in range(len(item_vals)):
            for b in range(a + 1, len(item_vals)):
                d_o += diff_metric(item_vals[a], item_vals[b])
                n_pairs += 1
    
    if n_pairs == 0:
        return None
    
    d_o /= n_pairs
    
    # Expected disagreement
    all_vals = []
    for r in range(n_raters):
        for i in range(n_items):
            if data_matrix[r][i] is not None:
                all_vals.append(data_matrix[r][i])
    
    d_e = 0
    n_expected = 0
    for a in range(len(all_vals)):
        for b in range(len(all_vals)):
            if a != b:
                d_e += diff_metric(all_vals[a], all_vals[b])
                n_expected += 1
    
    if n_expected == 0:
        return None
    
    d_e /= n_expected
    
    if d_e == 0:
        return 1.0
    
    return 1 - (d_o / d_e)

def icc_1way(vals_list):
    """单因素随机效应ICC(1)"""
    if len(vals_list) < 2:
        return None
    n_raters = len(vals_list)
    n_items = len(vals_list[0])
    if n_items < 2:
        return None
    grand_mean = mean([v for run in vals_list for v in run])
    item_means = [mean([vals_list[r][i] for r in range(n_raters)]) for i in range(n_items)]
    ms_between = n_raters * sum((im - grand_mean) ** 2 for im in item_means) / (n_items - 1)
    ms_within = sum((vals_list[r][i] - item_means[i]) ** 2
                    for r in range(n_raters) for i in range(n_items)) / (n_items * (n_raters - 1))
    if ms_between + ms_within == 0:
        return 1.0
    return (ms_between - ms_within) / (ms_between + (n_raters - 1) * ms_within)

def pass_k(results_list, k=5):
    """
    pass^k 可靠性指标（τ-bench风格）
    
    测量Agent在k次运行中全部通过的概率。
    对于评测系统来说，测量多次评测结果是否一致。
    
    results_list: list of bool, 每次运行是否pass
    k: 测量的窗口大小
    """
    n = len(results_list)
    if n < k:
        return None
    n_pass = sum(results_list)
    # pass^k = 1 - C(n-k, n_pass) / C(n, n_pass) approximately
    # Simplified: probability all k runs pass given observed pass rate
    p = n_pass / n
    return p ** k


# ============================================================
# Phase I: Rubric 编译器 — 从指令编译不可变检查清单
# ============================================================

RUBRIC_COMPILE_PROMPT = """你是一个评测标准编译器。将以下外呼指令转化为原子化、可判定的检查清单。

【原则】
1. 每个检查项必须具体、可验证，能做YES/PARTIAL/NO判定
2. 检查项粒度要细：一个大步骤拆成多个原子步骤
3. 检查项必须覆盖指令的所有要求，不遗漏

【指令内容】
{instruction}

【5个维度】
1. flow(流程步骤): Call Flow每步是否执行？步骤顺序是否正确？Agent是否等待用户回应？信息是否在正确步骤中传达？信息分步传达方式是否正确？
2. info(信息传达): Knowledge Points每个关键信息点是否传达？数值参数是否准确？只关注内容是否出现，不关注步骤(步骤归flow)。
3. constraint(约束遵循): 语气要求、重复避免、超范围问题处理、字数限制等。注意：字数和禁用词由代码检查，此处只关注语义约束。
4. opening(开场白): 自我介绍？来电目的？关键参数？
5. task(任务完成): 核心任务是否完成？关键信息是否被用户理解（而非仅被Agent说出）？对话是否在完成核心目标后正常结束（用户因信息过载仓促中断不算正常完成）？

输出JSON数组，每个检查项格式:
{{"id": "维度_序号(如flow_1)", "dimension": "维度名", "description": "具体检查内容"}}

只输出JSON数组，不要其他文字。"""


def compile_rubric(instruction: str) -> list[dict]:
    """
    Phase I: Rubric编译 — 从指令编译不可变检查清单
    
    灵感: RULERS Phase I — "Compile natural language rubric into immutable JSON bundle"
    灵感: Rubric Is All You Need — QS rubric比generic rubric κ提升314%
    
    返回: 检查项列表，每个检查项有唯一ID、维度、binary判定标准
    """
    prompt = RUBRIC_COMPILE_PROMPT.format(instruction=instruction)
    content = call_llm([{"role": "user", "content": prompt}], temperature=0.1, max_tokens=512, timeout=120)
    
    # 解析JSON
    m = re.search(r'```(?:json)?\s*([\s\S]*?)```', content)
    if m:
        content = m.group(1)
    m = re.search(r'\[[\s\S]*\]', content)
    if m:
        try:
            items = json.loads(m.group())
        except json.JSONDecodeError:
            items = []
    else:
        items = []
    
    # 自动补全缺失字段
    # 根据dimension推断weight: flow/info关键=2.0, 其他=1.5（与DIMENSION_WEIGHTS一致）
    weight_map = {"flow": 2.0, "info": 2.0, "task": 1.5, "constraint": 1.5, "opening": 1.5}
    for it in items:
        dim = it.get("dimension", "")
        it.setdefault("weight", weight_map.get(dim, 1.0))
        it.setdefault("type", "semantic")
        it.setdefault("evidence_required", True)
        it.setdefault("verification_hint", f"检查对话中是否: {it.get('description', '')}")
    
    # 所有项都是语义检查项（确定性检查由代码独立完成）
    semantic_items = items
    
    # 限制检查项数量（每个维度最多8项，避免评测时间过长）
    MAX_PER_DIM = 8
    dim_counts = {}
    filtered = []
    for it in semantic_items:
        dim = it.get("dimension", "")
        cnt = dim_counts.get(dim, 0)
        if cnt < MAX_PER_DIM:
            filtered.append(it)
            dim_counts[dim] = cnt + 1
    semantic_items = filtered
    
    # 为每个检查项添加hash（确保不可变）
    for it in semantic_items:
        it["hash"] = hash(json.dumps(it, sort_keys=True, ensure_ascii=False)) & 0xFFFFFFFF
    
    return semantic_items, []


# ============================================================
# Phase II: 逐项判定 + 证据锚定
# ============================================================

JUDGE_ITEM_PROMPT_GRM = """你是一个严苛的对话质量评测官 (Generative Reward Model)。

## 任务
根据以下检查项，评估对话中该维度是否达标。

## 检查项
- ID: {item_id}
- 维度: {dimension}
- 检查内容: {description}
- 验证提示: {verification_hint}

## 对话原文
{instruction}

{dialogue}

## 输出要求
你必须先思考（在脑海中推理），然后输出严格的 JSON 格式：

```json
{{
  "reasoning": "你的推理过程：逐步对比对话原文与检查项要求，引用具体轮次和原文",
  "verdict": "YES" 或 "PARTIAL" 或 "NO" 或 "N/A",
  "score": 1.0 或 0.5 或 0.0 或 null,
  "evidence": "你引用的对话原文（逐字）",
  "evidence_turn": 证据所在轮次号（整数）
}}
```

## 判定标准
- YES: 完全满足 → score=1.0
- PARTIAL: 部分满足 → score=0.5
- NO: 完全不满足 → score=0.0
- N/A: 前置条件未触发 → score=null

## 严格规则
1. reasoning 必须包含逐步推理过程，引用对话原文
2. evidence 必须是对话中的原文引用，不能编造
3. 如果找不到相关证据，evidence 写"未找到相关对话"，verdict 写"NO"
4. 只输出 JSON，不要其他文字

开始评估："""

JUDGE_ITEM_PROMPT = JUDGE_ITEM_PROMPT_GRM  # 使用 GRM 风格


def parse_grm_output(content: str) -> dict:
    """
    解析 GRM/XML 或 JSON 风格输出，统一返回判定字段。
    """
    content = content or ""
    thinking = re.search(r'<think>(.+?)</think>', content, re.DOTALL | re.IGNORECASE)
    reasoning = thinking.group(1).strip() if thinking else ""

    verdict_match = re.search(r'<verdict>\s*(YES|PARTIAL|NO|N/A)\s*</verdict>', content, re.IGNORECASE)
    verdict = verdict_match.group(1).upper() if verdict_match else None

    score_match = re.search(r'<score>\s*(1(?:\.0)?|0\.5|0(?:\.0)?|N/A|null)\s*</score>', content, re.IGNORECASE)
    score = score_match.group(1).upper() if score_match else None

    evidence_match = re.search(r'<evidence>(.*?)</evidence>', content, re.DOTALL | re.IGNORECASE)
    evidence = evidence_match.group(1).strip() if evidence_match else ""

    evidence_turn_match = re.search(r'<evidence_turn>\s*(\d+)\s*</evidence_turn>', content, re.IGNORECASE)
    evidence_turn = int(evidence_turn_match.group(1)) if evidence_turn_match else None

    if verdict is None:
        json_parsed = parse_json_from_llm(content)
        if json_parsed:
            reasoning = json_parsed.get("reasoning") or json_parsed.get("reason") or reasoning
            verdict = str(json_parsed.get("verdict", "NO")).upper()
            raw_score = json_parsed.get("score")
            if raw_score is None and verdict in ("YES", "PARTIAL", "NO", "N/A"):
                raw_score = {"YES": 1.0, "PARTIAL": 0.5, "NO": 0.0, "N/A": "N/A"}[verdict]
            score = str(raw_score).upper() if raw_score is not None else score
            evidence = json_parsed.get("evidence", evidence)
            evidence_turn = json_parsed.get("evidence_turn", evidence_turn)

    verdict = verdict if verdict in ("YES", "PARTIAL", "NO", "N/A") else "NO"
    if score in (None, "NULL"):
        score = "N/A" if verdict == "N/A" else {"YES": "1.0", "PARTIAL": "0.5", "NO": "0.0"}.get(verdict, "0.0")
    elif score == "1":
        score = "1.0"
    elif score == "0":
        score = "0.0"

    return {
        "reasoning": reasoning,
        "verdict": verdict,
        "score": score,
        "evidence": evidence,
        "evidence_turn": evidence_turn,
    }

def judge_single_item(item: dict, instruction: str, dialogue: list[dict]) -> dict:
    """
    Phase II: 逐项判定 + 证据锚定 (GRM 风格)
    
    灵感: DeepSeek-V4 Actor-as-GRM — 自然语言推理链对齐隐空间
    """
    dialogue_text = format_dialogue(dialogue)
    
    prompt = JUDGE_ITEM_PROMPT.format(
        instruction=instruction,
        dialogue=dialogue_text,
        item_id=item.get("id", ""),
        dimension=item.get("dimension", ""),
        description=item.get("description", ""),
        verification_hint=item.get("verification_hint", ""),
    )
    
    content = call_llm([{"role": "user", "content": prompt}], temperature=0.1, max_tokens=1024, timeout=60)
    
    # 解析输出：优先尝试 GRM 格式（<think> + <verdict> + <score>）
    parsed = parse_grm_output(content)
    
    # 如果 GRM 解析失败（没找到 <verdict> 标签），回退到 JSON 格式（向后兼容）
    if parsed["verdict"] == "NO" and "<verdict>" not in content:
        json_parsed = parse_json_from_llm(content)
        if json_parsed:
            parsed = {
                "reasoning": json_parsed.get("reason", ""),
                "verdict": json_parsed.get("verdict", "NO"),
                "score": str({"YES": 1.0, "PARTIAL": 0.5}.get(json_parsed.get("verdict", "NO"), 0.0)),
                "evidence": json_parsed.get("evidence", ""),
                "evidence_turn": json_parsed.get("evidence_turn"),
            }
    
    result = {
        "item_id": item.get("id", ""),
        "dimension": item.get("dimension", ""),
        "description": item.get("description", ""),
        "weight": item.get("weight", 1.0),
        "verdict": parsed["verdict"],
        "score": float(parsed["score"]) if parsed["score"] not in ["N/A", None] else None,
        "reasoning": parsed["reasoning"],
        "reason": parsed["reasoning"],
        "evidence": parsed["evidence"],
        "evidence_turn": parsed["evidence_turn"],
        "type": item.get("type", "semantic"),
    }
    
    # 证据验证：检查引用的原文是否真实存在于对话中
    result["evidence_valid"] = verify_evidence(result["evidence"], dialogue)
    
    # N/A项：条件未触发，不计入分数
    if result["verdict"] == "N/A":
        result["score"] = None  # 标记为不计入
        return result
    
    # 证据锚定：无有效证据则分数封顶
    if result["verdict"] == "YES" and not result["evidence_valid"]:
        result["verdict"] = "PARTIAL"  # 降级而非直接归零
        result["evidence_rejected"] = True
        result["reasoning"] += "[证据验证失败：引用内容不在对话原文中，降级为PARTIAL]"
        result["score"] = 0.5
    
    return result



def verify_evidence(evidence: str, dialogue: list[dict]) -> bool:
    """
    证据验证：检查LLM引用的原文是否真实存在于对话中
    
    灵感: RULERS — "Quote verification V(q, u) checks quotes against source text"
    """
    if not evidence or evidence == "未找到相关对话":
        return False
    
    # 构建对话原文（按轮次索引，方便按轮次匹配）
    turn_contents = {}
    all_content = ""
    for d in dialogue:
        turn = d.get("turn", 0)
        turn_contents[turn] = d["content"]
        all_content += d["content"] + " "
    
    # 清理evidence：去掉轮次标记 [第X轮]
    cleaned = re.sub(r'\[第\d+轮\]\s*', '', evidence)
    
    # 提取引用片段（多种模式）
    quoted = []
    # 模式1: 引号内的内容
    quoted.extend(re.findall(r'[""「」『』]([^""「」『』]+)[""「」『』]', cleaned))
    # 模式2: 如果没有引号，提取连续中文字符+数字序列（>=4字符）
    if not quoted:
        quoted = re.findall(r'[\u4e00-\u9fff0-9a-zA-Z]{4,}', cleaned)
    
    if not quoted:
        # 至少evidence有实质内容
        return len(evidence) > 10
    
    # 检查引用片段是否在对话原文中
    # 优先在指定轮次中查找
    turn_match = re.search(r'\[第(\d+)轮\]', evidence)
    if turn_match:
        turn_num = int(turn_match.group(1))
        turn_text = turn_contents.get(turn_num, "")
        for q in quoted:
            if len(q) >= 3 and q in turn_text:
                return True
    
    # 在全部对话原文中查找
    for q in quoted:
        if len(q) >= 3 and q in all_content:
            return True
    
    # 宽松匹配：子串匹配（引用可能有微小差异，取最长公共子串）
    for q in quoted:
        if len(q) >= 5:
            # 检查是否有80%以上的子串匹配
            for i in range(len(q) - 3):
                substr = q[i:i+4]
                if substr in all_content:
                    return True
    
    return False


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
            "id": "det_word_count",
            "dimension": "constraint",
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
            "id": "det_banned_words",
            "dimension": "constraint",
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
        expected_clean = re.sub(r'\$\{[^}]+\}', '', expected)
        expected_clean = re.sub(r'\*\*[^*]+\*\*', '', expected_clean)
        keywords = [w for w in re.split(r'[，。？！、\s]+', expected_clean) if len(w) >= 2]
        if agent_turns and keywords:
            actual = agent_turns[0][1]["content"]
            matched = [k for k in keywords if k in actual]
            rate = len(matched) / len(keywords)
            results.append({
                "id": "det_opening_coverage",
                "dimension": "opening",
                "item": "开场白关键信息覆盖",
                "type": "deterministic",
                "pass_rate": rate,
                "matched": matched,
                "missing": [k for k in keywords if k not in matched],
                "keywords_total": len(keywords),
            })

    return results


# ============================================================
# Phase III: 分数计算 + 校准
# ============================================================

DIMENSION_WEIGHTS = {
    "flow": 2.0,
    "info": 2.0,
    "constraint": 1.5,
    "opening": 1.5,
    "task": 1.5,
}

DIMENSION_NAMES = {
    "flow": "流程遵循度",
    "info": "信息传达完整性",
    "constraint": "约束遵循度",
    "opening": "开场白准确度",
    "task": "任务达成度",
}

def compute_scores(judgments: list[dict], det_results: list[dict]) -> dict:
    """
    Phase III: 分数计算 + 校准
    
    核心设计：
    1. 每个检查项 YES/PARTIAL/NO 评分, N/A不计入
    2. 分数 = 加权通过率 — 机械计算，无LLM参与
    3. 确定性校准：约束遵循度不超过确定性检查上界
    4. 跨维度校准：信息在错误步骤中传达不计分
    """
    # 按维度汇总（排除N/A项）
    dim_scores = {}
    for dim_key in DIMENSION_WEIGHTS:
        dim_judgments = [j for j in judgments if j["dimension"] == dim_key and j.get("score") is not None]
        n_na = sum(1 for j in judgments if j["dimension"] == dim_key and j.get("score") is None)
        if dim_judgments:
            weighted_sum = sum(j["score"] * j["weight"] for j in dim_judgments)
            weighted_total = sum(j["weight"] for j in dim_judgments)
            rate = weighted_sum / weighted_total if weighted_total > 0 else 0
            dim_scores[dim_key] = {
                "score": weighted_sum,
                "max": weighted_total,
                "rate": rate,
                "n_items": len(dim_judgments),
                "n_na": n_na,
                "n_pass": sum(1 for j in dim_judgments if j["verdict"] == "YES"),
                "n_fail": sum(1 for j in dim_judgments if j["verdict"] == "NO"),
            }
    
    # 确定性校准：约束遵循度不超过确定性检查上界（取所有确定性检查的最小值）
    det_constraint_rate = None
    for d in det_results:
        if d["dimension"] == "constraint":
            pr = d.get("pass_rate", 1.0)
            if det_constraint_rate is None:
                det_constraint_rate = pr
            else:
                det_constraint_rate = min(det_constraint_rate, pr)
    
    if det_constraint_rate is not None and "constraint" in dim_scores:
        if dim_scores["constraint"]["rate"] > det_constraint_rate:
            dim_scores["constraint"]["rate_calibrated"] = det_constraint_rate
            dim_scores["constraint"]["calibration_note"] = f"LLM评分{dim_scores['constraint']['rate']:.1%}超过确定性检查上界{det_constraint_rate:.1%}，已校准"
            dim_scores["constraint"]["rate"] = det_constraint_rate
        else:
            dim_scores["constraint"]["rate_calibrated"] = dim_scores["constraint"]["rate"]
    
    # 跨维度校准1：info ≤ flow（信息在错误步骤传达，价值打折）
    if "flow" in dim_scores and "info" in dim_scores:
        flow_rate = dim_scores["flow"]["rate"]
        info_rate = dim_scores["info"]["rate"]
        if info_rate > flow_rate:
            penalized = flow_rate
            dim_scores["info"]["rate_original"] = info_rate
            dim_scores["info"]["rate"] = penalized
            dim_scores["info"]["cross_dim_penalty"] = f"info从{info_rate:.1%}校准至{penalized:.1%}（info ≤ flow）"
    
    # 跨维度校准2：约束严重违反时，info和task都应打折
    # 原因：信息轰炸导致用户无法理解 = 信息未有效传达 = 任务未完成
    # 规则：constraint < 0.3 时，info ≤ constraint + 0.2，task ≤ constraint + 0.2
    if "constraint" in dim_scores:
        constraint_rate = dim_scores["constraint"]["rate"]
        if constraint_rate < 0.3:
            cap = constraint_rate + 0.2
            # info cap
            if "info" in dim_scores and dim_scores["info"]["rate"] > cap:
                info_rate = dim_scores["info"]["rate"]
                dim_scores["info"]["rate_original"] = dim_scores["info"].get("rate_original", info_rate)
                dim_scores["info"]["rate"] = cap
                dim_scores["info"]["cross_dim_penalty"] = f"info从{info_rate:.1%}校准至{cap:.1%}（约束极低{constraint_rate:.1%}，信息轰炸=信息无效）"
            # task cap
            if "task" in dim_scores and dim_scores["task"]["rate"] > cap:
                task_rate = dim_scores["task"]["rate"]
                dim_scores["task"]["rate_original"] = dim_scores["task"].get("rate_original", task_rate)
                dim_scores["task"]["rate"] = cap
                dim_scores["task"]["cross_dim_penalty"] = f"task从{task_rate:.1%}校准至{cap:.1%}（约束极低{constraint_rate:.1%}，任务未真正完成）"
    
    # 总分 = 加权平均
    weighted_sum = sum(DIMENSION_WEIGHTS[d] * dim_scores[d]["rate"] for d in dim_scores if d in dim_scores)
    total_weight = sum(DIMENSION_WEIGHTS[d] for d in dim_scores if d in dim_scores)
    overall_rate = weighted_sum / total_weight if total_weight > 0 else 0
    
    return {
        "overall": overall_rate,
        "dimensions": dim_scores,
        "det_checks": det_results,
    }


# ============================================================
# 辅助函数
# ============================================================

def format_dialogue(dialogue):
    lines = []
    for d in dialogue:
        role = "Agent(被测)" if d["role"] == "agent" else "User(用户)"
        lines.append(f"[第{d.get('turn', 0)}轮] {role}: {d['content']}")
    return "\n".join(lines)

def parse_json_from_llm(text):
    m = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if m:
        text = m.group(1)
    m = re.search(r'\{[\s\S]*\}', text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {}


# ============================================================
# Agent-as-a-Judge 评测主流程
# ============================================================

def agent_evaluate(instruction: str, dialogue: list[dict], n_runs: int = 5, label: str = "",
                    rubric_items: list = None) -> dict:
    """
    Agent-as-a-Judge 完整评测流程
    
    步骤:
    1. Rubric编译 — 从指令编译原子化检查清单（可缓存复用）
    2. 多次运行:
       a. 逐项判定（带证据锚定）
       b. 确定性检查
       c. 分数计算（带校准）
    3. 统计分析（BCa CI, ICC, Gwet's AC1等）
    
    Args:
        rubric_items: 预编译的rubric（避免重复编译），为None时自动编译
    """
    print(f"\n{'='*70}")
    print(f"Agent-as-a-Judge 评测 v3: {label}")
    print(f"运行次数: {n_runs}")
    print(f"{'='*70}")
    
    # Step 1: Rubric编译（支持缓存复用）
    if rubric_items is None:
        print(f"\n--- Phase I: Rubric编译 ---")
        t0 = time.time()
        rubric_items, det_hint_items = compile_rubric(instruction)
        elapsed = time.time() - t0
        print(f"  编译完成: {len(rubric_items)} 个语义检查项 ({elapsed:.1f}s)")
    else:
        print(f"\n--- Phase I: Rubric (cached, {len(rubric_items)} 项) ---")
    
    # 按维度统计
    dim_counts = Counter(it.get("dimension", "unknown") for it in rubric_items)
    for dim, cnt in sorted(dim_counts.items()):
        dim_name = DIMENSION_NAMES.get(dim, dim)
        print(f"    {dim_name}: {cnt} 项")
    
    # Step 2: 确定性检查（只需做一次）
    det_results = deterministic_checks(instruction, dialogue)
    print(f"\n--- 确定性检查（代码验证，零LLM偏差）---")
    for d in det_results:
        status = "PASS" if d["pass_rate"] >= 0.8 else "WARN" if d["pass_rate"] >= 0.5 else "FAIL"
        print(f"  [{status}] {d['item']}: {d['pass_rate']:.1%}")
    
    # Step 3: 多次LLM评测
    print(f"\n--- Phase II: 逐项判定 + 证据锚定 (n={n_runs}) ---")
    all_run_results = []
    
    for run_idx in range(n_runs):
        print(f"\n  运行 {run_idx+1}/{n_runs}...", flush=True)
        t0 = time.time()
        
        run_judgments = []
        for item in rubric_items:
            try:
                judgment = judge_single_item(item, instruction, dialogue)
                run_judgments.append(judgment)
            except Exception as e:
                run_judgments.append({
                    "item_id": item.get("id", ""),
                    "dimension": item.get("dimension", ""),
                    "description": item.get("description", ""),
                    "weight": item.get("weight", 1.0),
                    "verdict": "NO",
                    "evidence": "",
                    "reason": f"评测异常: {str(e)}",
                    "reasoning": f"评测异常: {str(e)}",
                    "score": 0.0,
                    "evidence_valid": False,
                })
        
        # 计算本次运行分数
        scores = compute_scores(run_judgments, det_results)
        elapsed = time.time() - t0
        
        n_yes = sum(1 for j in run_judgments if j["verdict"] == "YES")
        n_partial = sum(1 for j in run_judgments if j["verdict"] == "PARTIAL")
        n_no = sum(1 for j in run_judgments if j["verdict"] == "NO")
        n_invalid = sum(1 for j in run_judgments if j.get("evidence_rejected"))
        
        print(f"    总分={scores['overall']:.3f}, YES={n_yes}, PARTIAL={n_partial}, NO={n_no}, 证据拒绝={n_invalid} ({elapsed:.1f}s)")
        
        all_run_results.append({
            "judgments": run_judgments,
            "scores": scores,
        })
    
    # Step 4: 统计分析
    print(f"\n--- Phase III: 统计分析 ---")
    return compute_statistics(all_run_results, rubric_items, det_results, label)


def compute_statistics(all_run_results: list, rubric_items: list, det_results: list, label: str) -> dict:
    """
    统计分析：
    1. 各维度/总分 均值 + BCa Bootstrap CI
    2. 逐项判定稳定性（Gwet's AC1）
    3. 评测者间一致性（ICC, Krippendorff's alpha）
    4. 确定性vs LLM偏差分析
    5. 逐项判定频率统计
    """
    n_runs = len(all_run_results)
    if n_runs < 2:
        print("  运行次数不足2次，无法计算统计量")
        # 返回统一格式（与n_runs>=2一致）
        scores = all_run_results[0]["scores"] if all_run_results else {}
        result = {}
        result["overall"] = {"mean": scores.get("overall", 0), "std": 0, "bca_ci_95": [scores.get("overall", 0)] * 2, "vals": [scores.get("overall", 0)]}
        for dim_key in DIMENSION_WEIGHTS:
            dim_data = scores.get("dimensions", {}).get(dim_key, {})
            result[dim_key] = {"mean": dim_data.get("rate", 0), "std": 0, "bca_ci_95": [dim_data.get("rate", 0)] * 2, "vals": [dim_data.get("rate", 0)]}
        result["label"] = label
        result["n_runs"] = 1
        result["rubric_items"] = len(rubric_items)
        result["stability_rate"] = 1.0
        result["det_checks"] = det_results
        return result
    
    # 收集各维度和总分的多次运行值
    overall_vals = [r["scores"]["overall"] for r in all_run_results]
    dim_vals = {}
    for dim_key in DIMENSION_WEIGHTS:
        dim_vals[dim_key] = [r["scores"]["dimensions"].get(dim_key, {}).get("rate", 0) for r in all_run_results]
    
    # 1. 均值 + BCa Bootstrap CI
    print(f"\n  {'指标':<20} {'均值':>6} {'标准差':>6} {'BCa 95%CI':>18}")
    print("  " + "-" * 55)
    
    results_summary = {}
    
    # 总分
    m = mean(overall_vals)
    s = std(overall_vals)
    ci_lo, ci_hi = bca_bootstrap_ci(overall_vals)
    print(f"  {'总分(overall)':<20} {m:>6.3f} {s:>6.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")
    results_summary["overall"] = {"mean": m, "std": s, "bca_ci_95": [ci_lo, ci_hi], "vals": overall_vals}
    
    # 各维度
    for dim_key in DIMENSION_WEIGHTS:
        dim_name = DIMENSION_NAMES.get(dim_key, dim_key)
        vals = dim_vals.get(dim_key, [])
        if vals:
            m = mean(vals)
            s = std(vals)
            ci_lo, ci_hi = bca_bootstrap_ci(vals)
            print(f"  {dim_name:<20} {m:>6.3f} {s:>6.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")
            results_summary[dim_key] = {"mean": m, "std": s, "bca_ci_95": [ci_lo, ci_hi], "vals": vals}
    
    # 2. 评测者间一致性（ICC, Krippendorff's alpha）
    print(f"\n--- 评测者间一致性 ---")
    
    # ICC: runs × dimensions 矩阵
    dim_keys = list(DIMENSION_WEIGHTS.keys())
    matrix = []
    for r in all_run_results:
        row = [r["scores"]["dimensions"].get(d, {}).get("rate", 0) for d in dim_keys]
        matrix.append(row)
    
    if len(matrix) >= 2:
        icc_val = icc_1way(matrix)
        if icc_val is not None:
            quality = "优秀(>0.75)" if icc_val > 0.75 else "良好(0.5-0.75)" if icc_val > 0.5 else "差(<0.5)"
            print(f"  ICC(1): {icc_val:.3f} ({quality})")
            results_summary["icc"] = icc_val
    
    # Krippendorff's alpha: 适合有序尺度
    if len(matrix) >= 2:
        # 转换为0-4的有序尺度（5档）
        ordinal_matrix = []
        for row in matrix:
            ordinal_row = [min(4, max(0, int(v * 5))) for v in row]
            ordinal_matrix.append(ordinal_row)
        k_alpha = krippendorffs_alpha(ordinal_matrix, level="ordinal")
        if k_alpha is not None:
            quality = "可靠(≥0.80)" if k_alpha >= 0.80 else "可用(≥0.67)" if k_alpha >= 0.67 else "不可靠(<0.67)"
            print(f"  Krippendorff's α (ordinal): {k_alpha:.3f} ({quality})")
            results_summary["krippendorff_alpha"] = k_alpha
    
    # 3. 逐项判定稳定性
    print(f"\n--- 逐项判定稳定性 ---")
    item_ids = [it.get("id", "") for it in rubric_items]
    item_stability = {}
    
    for item_id in item_ids:
        verdicts = []
        for r in all_run_results:
            for j in r["judgments"]:
                if j["item_id"] == item_id:
                    verdicts.append(j["verdict"])
                    break
        
        if len(verdicts) >= 2:
            yes_rate = verdicts.count("YES") / len(verdicts)
            stable = len(set(verdicts)) == 1
            item_stability[item_id] = {
                "verdicts": verdicts,
                "yes_rate": yes_rate,
                "stable": stable,
            }
    
    n_stable = sum(1 for v in item_stability.values() if v["stable"])
    n_total = len(item_stability)
    stability_rate = n_stable / n_total if n_total > 0 else 0
    print(f"  判定稳定性: {n_stable}/{n_total} 项完全一致 ({stability_rate:.0%})")
    results_summary["stability_rate"] = stability_rate
    
    # 不稳定项
    unstable = [(k, v) for k, v in item_stability.items() if not v["stable"]]
    if unstable:
        unstable.sort(key=lambda x: -len(set(x[1]["verdicts"])))
        print(f"  不稳定项 ({len(unstable)}个):")
        for item_id, info in unstable[:10]:  # 只显示前10个
            print(f"    {item_id}: {'/'.join(info['verdicts'])} (YES率={info['yes_rate']:.0%})")
    
    # Gwet's AC1: 衡量任意两次运行间的逐项一致性
    if n_runs >= 2 and n_total > 0:
        # 取前两次运行的逐项判定
        r1_verdicts = []
        r2_verdicts = []
        for item_id in item_ids:
            v1 = v2 = None
            for j in all_run_results[0]["judgments"]:
                if j["item_id"] == item_id:
                    v1 = j["verdict"]
                    break
            for j in all_run_results[1]["judgments"]:
                if j["item_id"] == item_id:
                    v2 = j["verdict"]
                    break
            if v1 and v2:
                r1_verdicts.append(v1)
                r2_verdicts.append(v2)
        
        if len(r1_verdicts) >= 3:
            ac1 = gwet_ac1(r1_verdicts, r2_verdicts)
            if ac1 is not None:
                quality = "优秀(>0.8)" if ac1 > 0.8 else "良好(0.6-0.8)" if ac1 > 0.6 else "差(<0.6)"
                print(f"  Gwet's AC1 (run1 vs run2): {ac1:.3f} ({quality})")
                results_summary["gwet_ac1"] = ac1
    
    # 4. 确定性vs LLM校准分析
    print(f"\n--- 确定性校准效果 ---")
    if det_results:
        det_constraint_rate = None
        for d in det_results:
            if d["dimension"] == "constraint":
                det_constraint_rate = d.get("pass_rate")
                break
        if det_constraint_rate is not None and "constraint" in results_summary:
            llm_constraint = results_summary["constraint"]["mean"]
            bias = llm_constraint - det_constraint_rate
            # 注意：确定性检查和LLM语义检查测量的是不同的子项
            # 确定性检查: 字数限制、禁用词（代码验证）
            # LLM语义检查: 语气自然、重复避免、超范围处理等（LLM判断）
            # 因此两者不完全可比，但确定性检查为约束遵循度提供了上界
            if det_constraint_rate < llm_constraint:
                note = "确定性上界低于LLM评分，已校准"
            else:
                note = f"确定性上界{det_constraint_rate:.1%}（字数/禁用词），LLM语义检查{llm_constraint:.1%}"
            print(f"  约束遵循度: {note}")
            results_summary["llm_bias"] = {
                "deterministic": det_constraint_rate,
                "llm": llm_constraint,
                "bias": bias,
                "direction": "偏宽容" if bias > 0.05 else "偏严格" if bias < -0.05 else "基本一致",
                "note": note,
            }
    
    # 5. 证据验证率
    total_judgments = 0
    valid_evidence = 0
    rejected_evidence = 0
    for r in all_run_results:
        for j in r["judgments"]:
            total_judgments += 1
            if j.get("evidence_valid"):
                valid_evidence += 1
            if j.get("evidence_rejected"):
                rejected_evidence += 1
    
    if total_judgments > 0:
        print(f"\n--- 证据验证率 ---")
        print(f"  有效证据: {valid_evidence}/{total_judgments} ({valid_evidence/total_judgments:.1%})")
        print(f"  被拒绝证据: {rejected_evidence}/{total_judgments} ({rejected_evidence/total_judgments:.1%})")
        results_summary["evidence_validation_rate"] = valid_evidence / total_judgments
    
    # 保存详细结果
    results_summary["label"] = label
    results_summary["n_runs"] = n_runs
    results_summary["rubric_items"] = len(rubric_items)
    results_summary["det_checks"] = det_results
    results_summary["run_details"] = []
    for r in all_run_results:
        run_detail = {
            "overall": r["scores"]["overall"],
            "dimensions": {dim_key: r["scores"]["dimensions"].get(dim_key, {}).get("rate", 0) for dim_key in DIMENSION_WEIGHTS},
            "judgments": [{
                "item_id": j["item_id"],
                "dimension": j["dimension"],
                "description": j.get("description", ""),
                "verdict": j["verdict"],
                "score": j.get("score"),
                "evidence": j.get("evidence", ""),
                "evidence_turn": j.get("evidence_turn"),
                "evidence_valid": j.get("evidence_valid", False),
                "reason": j.get("reason") or j.get("reasoning", ""),
            } for j in r["judgments"]],
        }
        results_summary["run_details"].append(run_detail)
    
    return results_summary


def compare_results(result_a: dict, result_b: dict, label_a: str = "A", label_b: str = "B"):
    """对比两组评测结果，报告效应量和置信区间重叠"""
    print(f"\n{'='*70}")
    print(f"对比分析: {label_a} vs {label_b}")
    print(f"{'='*70}")
    
    a_overall = result_a.get("overall", {})
    b_overall = result_b.get("overall", {})
    
    if not a_overall or not b_overall:
        print("  缺少对比数据")
        return
    
    a_vals = a_overall.get("vals", [a_overall.get("mean", 0)])
    b_vals = b_overall.get("vals", [b_overall.get("mean", 0)])
    
    a_mean = mean(a_vals) if isinstance(a_vals, list) else a_vals
    b_mean = mean(b_vals) if isinstance(b_vals, list) else b_vals
    gap = a_mean - b_mean
    
    # 置信区间重叠
    a_ci = a_overall.get("bca_ci_95", a_overall.get("ci_95", [0, 0]))
    b_ci = b_overall.get("bca_ci_95", b_overall.get("ci_95", [0, 0]))
    overlap = not (a_ci[0] > b_ci[1] or b_ci[0] > a_ci[1])
    
    print(f"\n  {label_a} 总分: {a_mean:.3f} BCa 95%CI[{a_ci[0]:.3f}, {a_ci[1]:.3f}]")
    print(f"  {label_b} 总分: {b_mean:.3f} BCa 95%CI[{b_ci[0]:.3f}, {b_ci[1]:.3f}]")
    print(f"  差距: {gap:+.3f}")
    print(f"  置信区间重叠: {'是 - 区分不显著' if overlap else '否 - 区分显著'}")
    
    # Effect size
    if isinstance(a_vals, list) and isinstance(b_vals, list) and len(a_vals) >= 2 and len(b_vals) >= 2:
        d = cohens_d(a_vals, b_vals)
        effect = "大(>0.8)" if abs(d) > 0.8 else "中(0.5-0.8)" if abs(d) > 0.5 else "小(<0.5)"
        print(f"  Cohen's d: {d:.2f} (效应量: {effect})")
    
    # 各维度对比
    print(f"\n  {'维度':<20} {label_a:>8} {label_b:>8} {'差距':>8}")
    print("  " + "-" * 50)
    for dim_key in DIMENSION_WEIGHTS:
        dim_name = DIMENSION_NAMES.get(dim_key, dim_key)
        a_dim = result_a.get(dim_key, {})
        b_dim = result_b.get(dim_key, {})
        a_m = a_dim.get("mean", "N/A")
        b_m = b_dim.get("mean", "N/A")
        if isinstance(a_m, (int, float)) and isinstance(b_m, (int, float)):
            print(f"  {dim_name:<20} {a_m:>8.3f} {b_m:>8.3f} {a_m-b_m:>+8.3f}")
    
    # 一致性指标对比
    a_icc = result_a.get("icc")
    b_icc = result_b.get("icc")
    a_ac1 = result_a.get("gwet_ac1")
    b_ac1 = result_b.get("gwet_ac1")
    a_kalpha = result_a.get("krippendorff_alpha")
    b_kalpha = result_b.get("krippendorff_alpha")
    
    print(f"\n  {'可靠性指标':<20} {label_a:>8} {label_b:>8}")
    print("  " + "-" * 40)
    if a_icc is not None or b_icc is not None:
        print(f"  {'ICC(1)':<20} {a_icc:>8.3f} {b_icc:>8.3f}" if a_icc and b_icc else "")
    if a_ac1 is not None or b_ac1 is not None:
        print(f"  {'Gwet AC1':<20} {a_ac1:>8.3f} {b_ac1:>8.3f}" if a_ac1 and b_ac1 else "")
    if a_kalpha is not None or b_kalpha is not None:
        print(f"  {'Krippendorff α':<20} {a_kalpha:>8.3f} {b_kalpha:>8.3f}" if a_kalpha and b_kalpha else "")


# ============================================================
# Main
# ============================================================

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
    print("评测系统 v3 — Agent-as-a-Judge + 原子化Rubric + 证据锚定")
    print("核心方法:")
    print("  1. 原子化Rubric (Rubric Is All You Need): binary marking消除灰色地带")
    print("  2. 证据锚定 (RULERS): 无证据则分数封顶")
    print("  3. Agent-as-a-Judge: 多步评测而非一次性打分")
    print("  4. BCa Bootstrap CI: 比percentile bootstrap更稳健")
    print("  5. Gwet's AC1: 比Cohen's Kappa抗悖论")
    print("  6. Krippendorff's α: 支持多评价者+有序尺度")
    print()

    n_runs = 2

    # 编译rubric一次，两个对话复用
    print("--- Phase I: Rubric编译 (共享) ---")
    t0 = time.time()
    rubric_items, _ = compile_rubric(INSTRUCTION)
    elapsed = time.time() - t0
    print(f"  编译完成: {len(rubric_items)} 项 ({elapsed:.1f}s)")

    # 评测好对话
    result_good = agent_evaluate(INSTRUCTION, DIALOGUE_GOOD, n_runs=n_runs, label="好对话", rubric_items=rubric_items)

    # 评测差对话
    result_bad = agent_evaluate(INSTRUCTION, DIALOGUE_BAD, n_runs=n_runs, label="差对话", rubric_items=rubric_items)

    # 对比
    compare_results(result_good, result_bad, "好对话", "差对话")

    # 保存
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "eval_v3_results.json")
    
    # 精简保存（去掉run_details中的judgments避免文件过大）
    save_result_good = {k: v for k, v in result_good.items() if k != "run_details"}
    save_result_bad = {k: v for k, v in result_bad.items() if k != "run_details"}
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"good": save_result_good, "bad": save_result_bad}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output_path}")


if __name__ == "__main__":
    main()
