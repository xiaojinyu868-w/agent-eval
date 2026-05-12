"""
评测系统 v3 完整运行入口

整合:
1. eval_v3.py — Agent-as-a-Judge + 原子化Rubric + 证据锚定
2. user_simulator_v2.py — 参数化Persona + Sim2Real Gap缓解
3. 高级统计 — BCa CI, Gwet's AC1, Krippendorff's α

完整流程:
1. 生成多样化对话 (user_simulator_v2)
2. 对每条对话运行Agent-as-a-Judge评测 (eval_v3)
3. 跨对话统计分析
4. 输出可解释、可量化的评测报告

比赛要求: "自动产出评测报告，要求评测过程可解释结果可量化"
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(__file__))

from eval_v3 import (
    agent_evaluate, compare_results, compute_scores,
    deterministic_checks, compile_rubric,
    mean, std, bca_bootstrap_ci, icc_1way, krippendorffs_alpha,
    cohens_d, format_dialogue, DIMENSION_NAMES, DIMENSION_WEIGHTS,
)
from user_simulator_v2 import (
    simulate_dialogue, generate_diverse_dialogues,
    PRESET_PERSONAS, PersonaConfig, generate_random_persona,
)


def run_full_evaluation(instruction: str, dialogues: list[dict], n_runs: int = 5):
    """
    完整评测流程
    
    输入:
    - instruction: 任务指令
    - dialogues: [{"persona": {...}, "dialogue": [...], "persona_type": "..."}]
    - n_runs: 每条对话评测次数
    
    输出: 完整评测报告
    """
    print("=" * 70)
    print("评测系统 v3 完整流程")
    print(f"对话数: {len(dialogues)}, 每条评测次数: {n_runs}")
    print("=" * 70)
    
    # Step 0: 编译rubric一次，所有对话共享
    print("\n--- Phase 0: Rubric编译 (共享) ---")
    t0 = time.time()
    rubric_items, _ = compile_rubric(instruction)
    elapsed = time.time() - t0
    print(f"  编译完成: {len(rubric_items)} 项 ({elapsed:.1f}s)")
    
    # Step 1: 对每条对话运行评测（复用rubric）
    results = {}
    for i, dl in enumerate(dialogues):
        persona_type = dl.get("persona_type", f"persona_{i}")
        persona_info = dl.get("persona", {})
        persona_name = persona_info.get("name", persona_type)
        dialogue = dl["dialogue"]
        dialogue_id = f"dlg_{i:02d}_{persona_type}"
        
        print(f"\n{'='*70}")
        print(f"评测: {dialogue_id} ({persona_name}, {len(dialogue)}轮)")
        print(f"{'='*70}")
        
        t0 = time.time()
        result = agent_evaluate(instruction, dialogue, n_runs=n_runs, label=dialogue_id, rubric_items=rubric_items)
        elapsed = time.time() - t0
        
        if result:
            results[dialogue_id] = result
            overall = result.get("overall", {})
            print(f"\n  总分: {overall.get('mean', 0):.3f} ± {overall.get('std', 0):.3f}")
            ci = overall.get("bca_ci_95", [0, 0])
            print(f"  BCa 95%CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
            print(f"  耗时: {elapsed:.1f}s")
        else:
            print(f"  评测失败")
    
    # Step 2: 跨对话统计分析
    cross_dialogue_stats = compute_cross_dialogue_stats(results)
    
    # Step 3: 生成报告
    report = generate_report(instruction, results, cross_dialogue_stats, dialogues)
    
    return results, cross_dialogue_stats, report


def compute_cross_dialogue_stats(results: dict) -> dict:
    """跨对话统计分析"""
    print(f"\n{'='*70}")
    print("跨对话统计分析")
    print(f"{'='*70}")
    
    dialogue_ids = list(results.keys())
    if len(dialogue_ids) < 2:
        print("对话数不足2条，无法做跨对话统计")
        return {}
    
    # 1. 总分排名
    overall_means = [(did, results[did].get("overall", {}).get("mean", 0)) for did in dialogue_ids]
    overall_means.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n--- 总分排名 ---")
    for rank, (did, score) in enumerate(overall_means, 1):
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {rank}. {did:<30} {score:.3f} {bar}")
    
    # 2. 各维度跨对话统计
    dim_keys = list(DIMENSION_WEIGHTS.keys())
    print(f"\n--- 跨对话维度统计 ---")
    print(f"  {'维度':<20} {'均值':>6} {'标准差':>6} {'最小':>6} {'最大':>6}")
    print("  " + "-" * 50)
    
    dim_stats = {}
    for dim_key in dim_keys:
        dim_name = DIMENSION_NAMES.get(dim_key, dim_key)
        dim_means = [results[did].get(dim_key, {}).get("mean", 0) for did in dialogue_ids]
        if dim_means:
            m = mean(dim_means)
            s = std(dim_means)
            mn = min(dim_means)
            mx = max(dim_means)
            print(f"  {dim_name:<20} {m:>6.3f} {s:>6.3f} {mn:>6.3f} {mx:>6.3f}")
            dim_stats[dim_key] = {"mean": m, "std": s, "min": mn, "max": mx}
    
    # 3. 评测系统可靠性指标
    print(f"\n--- 评测系统可靠性 ---")
    
    # 平均ICC
    icc_vals = [results[did].get("icc") for did in dialogue_ids if results[did].get("icc") is not None]
    if icc_vals:
        print(f"  平均 ICC(1): {mean(icc_vals):.3f}")
    
    # 平均Gwet's AC1
    ac1_vals = [results[did].get("gwet_ac1") for did in dialogue_ids if results[did].get("gwet_ac1") is not None]
    if ac1_vals:
        print(f"  平均 Gwet's AC1: {mean(ac1_vals):.3f}")
    
    # 平均Krippendorff's alpha
    kalpha_vals = [results[did].get("krippendorff_alpha") for did in dialogue_ids if results[did].get("krippendorff_alpha") is not None]
    if kalpha_vals:
        print(f"  平均 Krippendorff's α: {mean(kalpha_vals):.3f}")
    
    # 平均判定稳定性
    stability_vals = [results[did].get("stability_rate", 0) for did in dialogue_ids]
    print(f"  平均判定稳定性: {mean(stability_vals):.0%}")
    
    # 4. 确定性校准效果
    print(f"\n--- 确定性校准效果 ---")
    for did in dialogue_ids:
        llm_bias = results[did].get("llm_bias", {})
        if llm_bias:
            print(f"  {did}: 偏差={llm_bias.get('bias', 0):+.1%} ({llm_bias.get('direction', 'N/A')})")
    
    return {
        "ranking": overall_means,
        "dimension_stats": dim_stats,
        "avg_icc": mean(icc_vals) if icc_vals else None,
        "avg_gwet_ac1": mean(ac1_vals) if ac1_vals else None,
        "avg_krippendorff_alpha": mean(kalpha_vals) if kalpha_vals else None,
        "avg_stability": mean(stability_vals) if stability_vals else None,
    }


def generate_report(instruction: str, results: dict, cross_stats: dict, dialogues: list) -> str:
    """
    生成可解释、可量化的评测报告
    
    比赛要求: "自动产出评测报告，要求评测过程可解释结果可量化"
    """
    lines = []
    lines.append("=" * 70)
    lines.append("外呼任务对话模型指令遵循效果评测报告")
    lines.append("评测系统: Agent-as-a-Judge v3")
    lines.append("=" * 70)
    lines.append("")
    
    # 1. 评测方法说明
    lines.append("【评测方法】")
    lines.append("  1. 原子化Rubric编译: 从指令自动生成指令特定的检查清单")
    lines.append("     - 每个检查项为YES/PARTIAL/NO/N/A判定，消除主观灰色地带")
    lines.append("     - 条件未触发的检查项判N/A，不计入分数")
    lines.append("     - 灵感: 'Rubric Is All You Need' (Pathak et al. 2025)")
    lines.append("  2. 证据锚定: 每个判定必须附带对话原文证据")
    lines.append("     - 无有效证据则YES降级为PARTIAL")
    lines.append("     - 灵感: RULERS (Hong et al. 2026)")
    lines.append("  3. Agent-as-a-Judge: 多步评测而非一次性打分")
    lines.append("     - 灵感: Agent-as-a-Judge (Zhuge et al. 2024)")
    lines.append("  4. 确定性校准锚点: 字数/禁用词等代码验证与LLM判定分离")
    lines.append("  5. 跨维度校准:")
    lines.append("     - info ≤ flow (信息在错误步骤传达，价值打折)")
    lines.append("     - constraint极低时info/task打折 (信息轰炸=信息无效)")
    lines.append("  6. 统计严谨性:")
    lines.append("     - BCa Bootstrap 95%置信区间(比percentile bootstrap更稳健)")
    lines.append("     - Gwet's AC1(比Cohen's Kappa抗悖论)")
    lines.append("     - Krippendorff's α(支持多评价者+有序尺度)")
    lines.append("")
    
    # 2. 评测概况
    lines.append("【评测概况】")
    lines.append(f"  对话数: {len(results)}")
    lines.append(f"  每条评测次数: {list(results.values())[0].get('n_runs', 'N/A') if results else 'N/A'}")
    if cross_stats:
        lines.append(f"  评测者间一致性 ICC(1): {cross_stats.get('avg_icc', 'N/A')}")
        lines.append(f"  判定者间一致 Gwet's AC1: {cross_stats.get('avg_gwet_ac1', 'N/A')}")
        lines.append(f"  多评价者一致 Krippendorff's α: {cross_stats.get('avg_krippendorff_alpha', 'N/A')}")
        lines.append(f"  逐项判定稳定性: {cross_stats.get('avg_stability', 'N/A')}")
    lines.append("")
    
    # 3. 总分排名
    lines.append("【总分排名】")
    if cross_stats and cross_stats.get("ranking"):
        for rank, (did, score) in enumerate(cross_stats["ranking"], 1):
            lines.append(f"  {rank}. {did:<30} {score:.3f}")
    lines.append("")
    
    # 4. 各对话详细结果
    for did, result in results.items():
        lines.append(f"--- {did} ---")
        overall = result.get("overall", {})
        ci = overall.get("bca_ci_95", [0, 0])
        lines.append(f"  总分: {overall.get('mean', 0):.3f} ± {overall.get('std', 0):.3f}")
        lines.append(f"  BCa 95%CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
        
        # 各维度
        for dim_key in DIMENSION_WEIGHTS:
            dim_name = DIMENSION_NAMES.get(dim_key, dim_key)
            dim_data = result.get(dim_key, {})
            if dim_data:
                m = dim_data.get("mean", 0)
                s = dim_data.get("std", 0)
                lines.append(f"  {dim_name}: {m:.3f} ± {s:.3f}")
        
        # 确定性校准
        llm_bias = result.get("llm_bias", {})
        if llm_bias:
            lines.append(f"  LLM偏差: {llm_bias.get('direction', 'N/A')} (偏差={llm_bias.get('bias', 0):+.1%})")
        
        # 证据验证率
        evr = result.get("evidence_validation_rate")
        if evr is not None:
            lines.append(f"  证据验证率: {evr:.1%}")
        
        lines.append("")
    
    # 5. 跨对话维度统计
    if cross_stats and cross_stats.get("dimension_stats"):
        lines.append("【跨对话维度统计】")
        lines.append(f"  {'维度':<20} {'均值':>6} {'标准差':>6} {'最小':>6} {'最大':>6}")
        lines.append("  " + "-" * 50)
        for dim_key, stats in cross_stats["dimension_stats"].items():
            dim_name = DIMENSION_NAMES.get(dim_key, dim_key)
            lines.append(f"  {dim_name:<20} {stats['mean']:>6.3f} {stats['std']:>6.3f} {stats['min']:>6.3f} {stats['max']:>6.3f}")
        lines.append("")
    
    # 6. 方法论声明
    lines.append("【方法论声明】")
    lines.append("  本评测系统遵循以下科学原则:")
    lines.append("  1. LLM评分不是黄金标准 — 多次评测取统计量")
    lines.append("  2. 报告均值+置信区间 — 不报单次分数")
    lines.append("  3. 评测者间一致性(ICC/AC1/α)作为评测可靠性指标")
    lines.append("  4. 确定性检查与LLM判定分离 — 确定性检查提供校准上界")
    lines.append("  5. 证据锚定 — 每个判定附带原文证据，无证据则降级")
    lines.append("  6. YES/PARTIAL/NO/N/A四级判定 — N/A不计分避免误判")
    lines.append("  7. BCa Bootstrap CI — 校正偏差和偏度，比percentile更稳健")
    lines.append("  8. Gwet's AC1 — 在高一致性场景下比Cohen's Kappa更可靠")
    lines.append("  9. 跨维度校准 — 信息违规传达/约束极低时相关维度打折")
    lines.append("  10. 参数化Persona模拟 — 覆盖多样用户行为(Sim2Real Gap缓解)")
    lines.append("")
    
    report_text = "\n".join(lines)
    
    # 保存报告
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "eval_v3_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n报告已保存: {report_path}")
    
    return report_text


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


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="评测系统 v3 完整运行")
    parser.add_argument("--dialogues", type=str, default=None,
                        help="对话数据文件路径 (JSON)")
    parser.add_argument("--n-runs", type=int, default=5,
                        help="每条对话评测次数 (默认5)")
    parser.add_argument("--n-dialogues", type=int, default=8,
                        help="生成的对话数 (默认8)")
    parser.add_argument("--skip-generate", action="store_true",
                        help="跳过对话生成，使用已有对话数据")
    args = parser.parse_args()
    
    # Step 1: 加载或生成对话
    if args.dialogues and os.path.exists(args.dialogues):
        with open(args.dialogues, "r", encoding="utf-8") as f:
            data = json.load(f)
        instruction = data.get("instruction", INSTRUCTION)
        dialogues = data.get("dialogues", [])
        print(f"从文件加载 {len(dialogues)} 条对话: {args.dialogues}")
    elif args.skip_generate:
        # 使用之前生成的对话
        v2_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "simulated_dialogues_v2.json")
        v1_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "simulated_dialogues.json")
        if os.path.exists(v2_path):
            with open(v2_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            instruction = data.get("instruction", INSTRUCTION)
            dialogues = data.get("dialogues", [])
            print(f"从v2对话加载 {len(dialogues)} 条对话")
        elif os.path.exists(v1_path):
            with open(v1_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            instruction = data.get("instruction", INSTRUCTION)
            dialogues = data.get("dialogues", [])
            print(f"从v1对话加载 {len(dialogues)} 条对话")
        else:
            print("未找到对话数据文件，将生成新对话")
            dialogues_data = generate_diverse_dialogues(instruction, n_dialogues=args.n_dialogues)
            dialogues = dialogues_data
    else:
        # 生成新对话
        print(f"生成 {args.n_dialogues} 条多样化对话...")
        dialogues = generate_diverse_dialogues(INSTRUCTION, n_dialogues=args.n_dialogues)
        instruction = INSTRUCTION
        
        # 保存生成的对话
        output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
        os.makedirs(output_dir, exist_ok=True)
        save_data = {
            "instruction": instruction,
            "dialogues": [{
                "persona": d.get("persona", {}),
                "persona_type": d.get("persona_type", "unknown"),
                "dialogue": d.get("dialogue", []),
                "behavior_metrics": d.get("behavior_metrics", {}),
                "persona_consistency": d.get("persona_consistency", {}),
            } for d in dialogues],
        }
        save_path = os.path.join(output_dir, "simulated_dialogues_v2.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        print(f"对话数据已保存: {save_path}")
    
    # Step 2: 运行完整评测
    results, cross_stats, report = run_full_evaluation(
        instruction, dialogues, n_runs=args.n_runs
    )
    
    # Step 3: 保存完整结果
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存精简结果（去掉run_details避免文件过大）
    save_results = {}
    for did, r in results.items():
        save_results[did] = {k: v for k, v in r.items() if k not in ("run_details", "det_checks")}
    
    output_path = os.path.join(output_dir, "eval_v3_full_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "results": save_results,
            "cross_dialogue_stats": {k: v for k, v in cross_stats.items() if k != "ranking"},
        }, f, ensure_ascii=False, indent=2)
    print(f"\n完整结果已保存: {output_path}")
