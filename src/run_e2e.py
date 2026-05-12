"""
端到端评测：用户模拟器生成对话 → 严格评测
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(__file__))

from user_simulator import simulate_dialogue, PERSONAS
from strict_eval import strict_evaluate, deterministic_checks, format_dialogue


# ============================================================
# 指令
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
- 如需退出飞毛腿，必须在前一天 **晚上8点** 之前在 App 的"飞毛腿报名"中取消；次日生效。
- 连续完成 **7 天**多日合同，且每天完成 **3 单**，将获得额外奖励（例如，与单日合同相比每单多 +2 元）。

# Constraints
- 遵循对话流程和常见问题解答。
- 如被问及超出职责范围的问题，回复："我向同事确认后再回电给你。我现在能回答的先回答。"
- 保持语气随意，像打电话一样自然。
- 每次回复控制在**约 30 个字以内**。
- 避免重复回复；如需重申，请换种方式礼貌表达。
- 如果骑手坚持确实无法配送，安慰他们后挂断电话。"""


def main():
    print("=" * 60)
    print("端到端评测：用户模拟器 → 对话生成 → 严格评测")
    print("=" * 60)

    all_results = {}

    # 5种 persona 生成对话并评测
    personas = ["cooperative", "reluctant", "rejecting", "curious", "driving"]

    for persona_type in personas:
        persona_name = PERSONAS[persona_type]["name"]
        print(f"\n{'='*60}")
        print(f"Persona: {persona_type} ({persona_name})")
        print(f"{'='*60}")

        # 1. 生成对话
        print(f"\n--- 生成对话 ---")
        try:
            dialogue = simulate_dialogue(INSTRUCTION, persona_type, max_turns=14)
        except Exception as e:
            print(f"  对话生成失败: {e}")
            continue

        # 打印对话
        for d in dialogue:
            role = "Agent" if d["role"] == "agent" else "User"
            print(f"  [{d.get('turn',0)}] {role}: {d['content']}")

        # 保存对话
        dialogue_path = os.path.join(os.path.dirname(__file__), "..", "outputs", f"dialogue_{persona_type}.json")
        os.makedirs(os.path.dirname(dialogue_path), exist_ok=True)
        with open(dialogue_path, "w", encoding="utf-8") as f:
            json.dump({"persona": persona_type, "instruction": INSTRUCTION, "dialogue": dialogue}, f, ensure_ascii=False, indent=2)

        # 2. 严格评测 (3 runs, 平衡速度和统计可靠性)
        print(f"\n--- 严格评测 (n=3) ---")
        result = strict_evaluate(INSTRUCTION, dialogue, n_runs=3, label=f"{persona_type}({persona_name})")

        if result:
            all_results[persona_type] = {
                "persona_name": persona_name,
                "dialogue_turns": len(dialogue),
                "overall": result.get("overall"),
                "dimensions": {k: v for k, v in result.items() if k in ["流程遵循度", "信息传达完整性", "约束遵循度", "任务达成度"]},
                "icc": result.get("icc"),
                "llm_bias": result.get("llm_bias"),
            }
        else:
            all_results[persona_type] = {"error": "评测失败"}

    # 汇总对比
    print(f"\n\n{'='*70}")
    print("汇总对比")
    print(f"{'='*70}")
    print(f"{'Persona':<10} {'轮次':>4} {'总分':>8} {'流程':>6} {'信息':>6} {'约束':>6} {'任务':>6} {'ICC':>6}")
    print("-" * 70)

    for persona_type in personas:
        r = all_results.get(persona_type, {})
        if "error" in r:
            print(f"{persona_type:<10} {'--':>4} {'失败':>8}")
            continue
        overall = r.get("overall", {})
        o_mean = f"{overall['mean']:.3f}" if overall else "N/A"
        dims = r.get("dimensions", {})
        turns = r.get("dialogue_turns", "?")
        icc_val = r.get("icc")
        icc_str = f"{icc_val:.2f}" if icc_val is not None else "N/A"

        dim_strs = []
        for d in ["流程遵循度", "信息传达完整性", "约束遵循度", "任务达成度"]:
            dm = dims.get(d, {})
            dim_strs.append(f"{dm.get('mean', 0):.2f}" if dm else "N/A")

        print(f"{persona_type:<10} {turns:>4} {o_mean:>8} {dim_strs[0]:>6} {dim_strs[1]:>6} {dim_strs[2]:>6} {dim_strs[3]:>6} {icc_str:>6}")

    # 分析
    print(f"\n--- 分析 ---")
    scores = {}
    for pt, r in all_results.items():
        if "error" not in r and r.get("overall"):
            scores[pt] = r["overall"]["mean"]

    if scores:
        best = max(scores, key=scores.get)
        worst = min(scores, key=scores.get)
        print(f"  最高分: {best} ({PERSONAS[best]['name']}) = {scores[best]:.3f}")
        print(f"  最低分: {worst} ({PERSONAS[worst]['name']}) = {scores[worst]:.3f}")
        print(f"  分数范围: {scores[worst]:.3f} - {scores[best]:.3f}")
        print(f"  分数排序: {' > '.join(f'{k}={v:.3f}' for k, v in sorted(scores.items(), key=lambda x: -x[1]))}")

    # 保存
    output_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "e2e_eval_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output_path}")


if __name__ == "__main__":
    main()
