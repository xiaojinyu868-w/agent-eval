"""
快速评测 - 一次 LLM 调用产出全部结果，对比 thinking vs non-thinking
"""

import sys
import os
import json
import time
import socket

sys.path.insert(0, os.path.dirname(__file__))


# ============================================================
# LLM 调用（支持 thinking/非 thinking）
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

def call_llm(messages, thinking=False, max_tokens=8192, temperature=0.3):
    payload = json.dumps({
        "model": "kimi_k2d6",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"thinking": thinking} if not thinking else {"thinking": True, "preserve_thinking": True},
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
            # De-chunk if needed
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
            msg = data["choices"][0]["message"]
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
            # For thinking mode: if content is empty, the model may still be thinking
            # Try to extract structured result from reasoning if content is empty
            if not content and reasoning:
                # Check if reasoning contains the final JSON answer
                import re as _re
                json_match = _re.search(r'\{[\s\S]*"overall_score"[\s\S]*\}', reasoning)
                if json_match:
                    content = json_match.group()
            return {"content": content, "reasoning": reasoning, "raw": msg}
        except Exception as e:
            print(f"  IP {ip} failed: {e}")
            continue
        finally:
            s.close()
    raise RuntimeError("All IPs failed")


# ============================================================
# 评测 Prompt - 一次性产出结构化结果
# ============================================================

EVAL_PROMPT = """你是一个专业的对话质量评测专家。请仔细阅读任务指令和对话记录，然后逐项评估。

【任务指令】
{instruction}

【对话记录】
{dialogue}

请从以下4个维度逐项评估，每项给出分数和证据：

1. **流程遵循度**(权重2.0): Agent是否按指令中的对话流程执行？关键步骤是否覆盖？条件分支是否正确？
2. **信息传达完整性**(权重2.0): 指令要求传达的关键信息是否都已传达？参数是否准确？
3. **约束遵循度**(权重1.5): 字数限制、语气要求、重复避免、超范围问题处理等约束是否遵守？
4. **任务达成度**(权重1.5): 核心任务是否完成？对话是否正常结束？

对每个维度，输出：
- items: 该维度下的具体检查项列表，每项包含:
  - name: 检查项名称
  - verdict: "pass"/"partial"/"fail"
  - evidence: 对话中的原文证据（必须逐字引用，标注轮次）
  - reason: 判定理由

最后输出:
- dimension_scores: 各维度得分率(pass=1,partial=0.5,fail=0，加权平均)
- overall_score: 总得分率(各维度加权平均)

严格以JSON格式输出：
{{
  "dimensions": {{
    "流程遵循度": {{
      "items": [
        {{"name": "...", "verdict": "...", "evidence": "...", "reason": "..."}}
      ],
      "score": 0.0-1.0
    }},
    ...
  }},
  "overall_score": 0.0-1.0
}}"""


# ============================================================
# 示例数据
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


def format_dialogue(dialogue):
    lines = []
    for d in dialogue:
        role = "Agent(被测)" if d["role"] == "agent" else "User(用户)"
        lines.append(f"[第{d['turn']}轮] {role}: {d['content']}")
    return "\n".join(lines)


def parse_eval_result(text):
    """从LLM输出中解析评测结果"""
    import re
    # Try to find JSON
    m = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if m:
        text = m.group(1)
    m = re.search(r'\{[\s\S]*\}', text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def run_single_eval(dialogue, label, thinking=False):
    """跑一次评测"""
    dialogue_text = format_dialogue(dialogue)
    prompt = EVAL_PROMPT.format(instruction=INSTRUCTION, dialogue=dialogue_text)

    print(f"\n{'='*60}")
    print(f"评测: {label} (thinking={thinking})")
    print(f"{'='*60}")

    t0 = time.time()
    result = call_llm([{"role": "user", "content": prompt}], thinking=thinking, max_tokens=8192, temperature=0.1)
    elapsed = time.time() - t0

    content = result["content"]
    reasoning = result["reasoning"]

    if reasoning and thinking:
        print(f"[思考过程] {reasoning[:300]}...")

    parsed = parse_eval_result(content)
    if parsed:
        overall = parsed.get("overall_score", "N/A")
        dims = parsed.get("dimensions", {})
        print(f"\n>>> 总分: {overall}")
        for dim_name, dim_data in dims.items():
            score = dim_data.get("score", "N/A")
            items = dim_data.get("items", [])
            pass_count = sum(1 for i in items if i.get("verdict") == "pass")
            partial_count = sum(1 for i in items if i.get("verdict") == "partial")
            fail_count = sum(1 for i in items if i.get("verdict") == "fail")
            print(f"  {dim_name}: {score} (pass={pass_count}, partial={partial_count}, fail={fail_count})")

        # 输出详细证据
        print(f"\n--- 详细判定 ---")
        for dim_name, dim_data in dims.items():
            print(f"\n[{dim_name}]")
            for item in dim_data.get("items", []):
                icon = {"pass": "✅", "partial": "⚠️", "fail": "❌"}.get(item.get("verdict"), "❓")
                print(f"  {icon} {item.get('name','')}")
                print(f"     证据: {item.get('evidence','')}")
                print(f"     理由: {item.get('reason','')}")
    else:
        print(f"\n解析失败，原始输出: {content[:500]}")

    print(f"\n耗时: {elapsed:.1f}s")
    return {"label": label, "thinking": thinking, "parsed": parsed, "overall": parsed.get("overall_score") if parsed else None, "elapsed": elapsed}


def main():
    results = []

    # 四组对比: good/bad × thinking/not-thinking
    for dialogue, label in [(DIALOGUE_GOOD, "好对话"), (DIALOGUE_BAD, "差对话")]:
        for thinking in [False, True]:
            mode = "thinking" if thinking else "instant"
            r = run_single_eval(dialogue, f"{label}_{mode}", thinking=thinking)
            results.append(r)

    # 汇总对比
    print("\n" + "=" * 60)
    print("对比汇总")
    print("=" * 60)
    print(f"{'对话':<8} {'模式':<10} {'总分':<8} {'耗时':<8}")
    print("-" * 40)
    for r in results:
        overall = f"{r['overall']:.2f}" if isinstance(r['overall'], (int, float)) else "N/A"
        print(f"{r['label'].split('_')[0]:<8} {r['label'].split('_')[1] if '_' in r['label'] else '':<10} {overall:<8} {r['elapsed']:.1f}s")

    # 关键指标：好对话分数是否显著高于差对话
    print("\n--- 关键对比 ---")
    for mode in ["instant", "thinking"]:
        good_score = [r for r in results if "好对话" in r["label"] and mode in r["label"]]
        bad_score = [r for r in results if "差对话" in r["label"] and mode in r["label"]]
        if good_score and bad_score:
            g = good_score[0]["overall"]
            b = bad_score[0]["overall"]
            if isinstance(g, (int, float)) and isinstance(b, (int, float)):
                gap = g - b
                print(f"{mode}: 好对话={g:.2f}, 差对话={b:.2f}, 差距={gap:.2f} {'✓ 有效区分' if gap > 0.1 else '✗ 区分不足'}")
            else:
                print(f"{mode}: 解析失败，无法对比")

    # 保存
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    output = {
        "results": [{
            "label": r["label"],
            "thinking": r["thinking"],
            "overall_score": r["overall"],
            "elapsed": r["elapsed"],
            "dimensions": r["parsed"].get("dimensions") if r["parsed"] else None,
        } for r in results],
    }
    with open(os.path.join(output_dir, "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到 outputs/eval_results.json")


if __name__ == "__main__":
    main()
