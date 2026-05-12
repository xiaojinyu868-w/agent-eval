"""
评测系统主入口 - 端到端评测 Demo

用法:
    python run_eval.py

输出:
    - 评测分数（单值 score_rate + 各维度分值）
    - 可解释评测报告
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(__file__))

from eval_agent import EvaluationAgent, LLMClient, DialogueTurn, EvaluationResult


# ============================================================
# 示例对话数据
# ============================================================

SAMPLE_INSTRUCTION_1 = """# Role
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

SAMPLE_DIALOGUE_GOOD = [
    {"role": "agent", "content": "你好，请问是张三吗？我是站长。你已报名飞毛腿，午餐晚餐高峰期要上线。单日每天至少完成5单，多日每天至少完成3单。", "turn_id": 1},
    {"role": "user", "content": "哦，好的，合同生效了吗？", "turn_id": 2},
    {"role": "agent", "content": "生效了！你今天能开始配送吗？", "turn_id": 3},
    {"role": "user", "content": "可以的，不过多日合同要连续几天？", "turn_id": 4},
    {"role": "agent", "content": "多日合同要连续7天完成配送，不然会影响合同。", "turn_id": 5},
    {"role": "user", "content": "那如果我想退出怎么办？", "turn_id": 6},
    {"role": "agent", "content": "前一天晚上8点前在App的飞毛腿报名里取消就行。", "turn_id": 7},
    {"role": "user", "content": "好的，那安全方面有什么注意的吗？", "turn_id": 8},
    {"role": "agent", "content": "注意安全！减少拒单和超时，恶劣天气单多，有助于保住资格。加油！", "turn_id": 9},
    {"role": "user", "content": "行，谢谢站长！", "turn_id": 10},
    {"role": "agent", "content": "不客气，出发吧！", "turn_id": 11},
]

SAMPLE_DIALOGUE_BAD = [
    {"role": "agent", "content": "喂，张三吗？我是站长。你报了飞毛腿，单日每天至少完成5单，多日每天至少完成3单，午餐晚餐高峰期要上线知道吧？另外合同已经生效了，你今天能开始配送吗？多日合同需要连续7天完成配送否则会影响合同的，如果你无法连续配送7天你的名额可能会被他人占用的，目前很多骑手都在申请飞毛腿。如果你需要退出的话前一天晚上8点之前在App里取消就行，连续完成7天多日合同每天完成3单的话还有额外奖励每单多2元。好的，你还有其他问题吗？", "turn_id": 1},
    {"role": "user", "content": "太多了我记不住...", "turn_id": 2},
    {"role": "agent", "content": "好的好的，没关系我再说一遍，单日每天5单多日每天3单连续7天...", "turn_id": 3},
    {"role": "user", "content": "我有点事，能稍后再说吗？", "turn_id": 4},
    {"role": "agent", "content": "好的，那我就先不说了，嗯好的，哈哈，拜拜。", "turn_id": 5},
]


def run_evaluation(instruction: str, dialogue_data: list[dict], instruction_id: str = "1") -> EvaluationResult:
    """执行评测"""
    llm = LLMClient()
    agent = EvaluationAgent(llm)

    dialogue = [
        DialogueTurn(role=d["role"], content=d["content"], turn_id=d.get("turn_id", i+1))
        for i, d in enumerate(dialogue_data)
    ]

    result = agent.evaluate(instruction, dialogue, instruction_id)
    return result


def main():
    print("评测系统启动 - 使用 Kimi K2.6")
    print()

    # ---- 评测好对话 ----
    print("=" * 60)
    print("评测示例1：较好的对话")
    print("=" * 60)

    result_good = run_evaluation(SAMPLE_INSTRUCTION_1, SAMPLE_DIALOGUE_GOOD, instruction_id="1")
    print(result_good.report)

    print("\n【结构化分数（可优化目标）】")
    score_output = {
        "instruction_id": result_good.instruction_id,
        "total_score": result_good.total_score,
        "max_score": result_good.max_score,
        "score_rate": result_good.score_rate,
        "dimensions": result_good.dimension_scores,
    }
    print(json.dumps(score_output, ensure_ascii=False, indent=2))

    # ---- 评测差对话 ----
    print("\n" + "=" * 60)
    print("评测示例2：较差的对话")
    print("=" * 60)

    result_bad = run_evaluation(SAMPLE_INSTRUCTION_1, SAMPLE_DIALOGUE_BAD, instruction_id="1")
    print(result_bad.report)

    print("\n【结构化分数（可优化目标）】")
    score_output = {
        "instruction_id": result_bad.instruction_id,
        "total_score": result_bad.total_score,
        "max_score": result_bad.max_score,
        "score_rate": result_bad.score_rate,
        "dimensions": result_bad.dimension_scores,
    }
    print(json.dumps(score_output, ensure_ascii=False, indent=2))

    # ---- 保存 ----
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "good_dialogue": {
            "score_rate": result_good.score_rate,
            "dimension_rates": {k: v["rate"] for k, v in result_good.dimension_scores.items()},
        },
        "bad_dialogue": {
            "score_rate": result_bad.score_rate,
            "dimension_rates": {k: v["rate"] for k, v in result_bad.dimension_scores.items()},
        },
    }

    output_path = os.path.join(output_dir, "eval_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
