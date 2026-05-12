"""
人类标注工具 - 收集人工评分作为黄金标准

使用方式:
    python human_annotate.py

输出:
    annotations/{dialogue_id}.json — 每条对话的人工标注
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))

DIALOGUES_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "simulated_dialogues.json")
ANNOTATIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "annotations")

DIMENSIONS = [
    ("流程遵循度", "Agent是否按Call Flow步骤执行？条件分支是否正确？分步传达还是一次性倾倒？"),
    ("信息传达完整性", "Knowledge Points中的关键信息是否传达？参数准确？传达方式是否正确？"),
    ("约束遵循度", "字数、语气、重复避免、超范围处理、特殊情况处理等约束是否遵守？"),
    ("任务达成度", "核心任务是否完成？对话正常结束？用户是否理解了关键信息？"),
]


def load_dialogues():
    with open(DIALOGUES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def annotate_dialogue(dialogue_data, instruction, dialogue_id):
    """交互式标注一条对话"""
    print("\n" + "=" * 60)
    print(f"对话ID: {dialogue_id}")
    print(f"Persona: {dialogue_data['persona']}")
    print("=" * 60)

    print("\n【任务指令摘要】")
    # 简要展示指令
    for line in instruction.split("\n")[:15]:
        print(f"  {line}")
    print("  ...")

    print("\n【对话记录】")
    for d in dialogue_data["dialogue"]:
        role = "Agent" if d["role"] == "agent" else "User"
        print(f"  [{d.get('turn', '?')}] {role}: {d['content']}")

    print("\n" + "-" * 60)
    print("请逐维度评分 (1=很差, 2=差, 3=一般, 4=好, 5=很好)")
    print("-" * 60)

    scores = {}
    comments = {}

    for dim_name, dim_desc in DIMENSIONS:
        while True:
            try:
                score = int(input(f"\n  {dim_name} ({dim_desc})\n  评分(1-5): "))
                if 1 <= score <= 5:
                    scores[dim_name] = score
                    break
                print("  请输入1-5")
            except ValueError:
                print("  请输入数字")
                continue

        comment = input(f"  备注(可选,回车跳过): ").strip()
        if comment:
            comments[dim_name] = comment

    overall = input(f"\n  整体评价(1-5): ")
    try:
        scores["overall"] = int(overall)
    except ValueError:
        scores["overall"] = sum(scores.values()) / len(scores)

    general_comment = input("  总体备注(可选): ").strip()

    annotation = {
        "dialogue_id": dialogue_id,
        "persona": dialogue_data["persona"],
        "scores": scores,
        "comments": comments,
        "general_comment": general_comment,
    }

    return annotation


def main():
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

    data = load_dialogues()
    instruction = data["instruction"]
    dialogues = data["dialogues"]

    print(f"共 {len(dialogues)} 条对话待标注")
    print(f"标注结果保存到: {ANNOTATIONS_DIR}")

    for i, dl in enumerate(dialogues):
        dialogue_id = f"dlg_{i:02d}_{dl['persona']}"
        output_path = os.path.join(ANNOTATIONS_DIR, f"{dialogue_id}.json")

        if os.path.exists(output_path):
            print(f"\n跳过已标注: {dialogue_id}")
            continue

        annotation = annotate_dialogue(dl, instruction, dialogue_id)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(annotation, f, ensure_ascii=False, indent=2)
        print(f"已保存: {output_path}")

    print("\n标注完成！")


if __name__ == "__main__":
    main()
