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
from instruction_loader import load_instruction_records, filter_instruction_records


def safe_slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text))[:80] or "task"


def normalize_dialogue_payload(data, fallback_instruction: str):
    """兼容 v1 list、v2 dict、单条 dialogue dict 三种历史格式。"""
    if isinstance(data, list):
        return fallback_instruction, data
    if not isinstance(data, dict):
        return fallback_instruction, []
    instruction = data.get("instruction", fallback_instruction)
    if isinstance(data.get("dialogues"), list):
        return instruction, data["dialogues"]
    if isinstance(data.get("dialogue"), list):
        return instruction, [{
            "persona": data.get("persona", {}),
            "persona_type": data.get("persona_type", data.get("persona", "loaded")),
            "dialogue": data["dialogue"],
            "behavior_metrics": data.get("behavior_metrics", {}),
            "persona_consistency": data.get("persona_consistency", {}),
            "simulator_quality": data.get("simulator_quality", {}),
        }]
    return instruction, []


def summarize_simulation_quality(dialogues: list[dict]) -> dict:
    metrics = [d.get("behavior_metrics", {}) for d in dialogues if d.get("behavior_metrics")]
    qualities = [d.get("simulator_quality", {}) for d in dialogues if d.get("simulator_quality")]
    consistencies = [d.get("persona_consistency", {}) for d in dialogues if d.get("persona_consistency")]
    if not dialogues:
        return {}
    return {
        "dialogues": len(dialogues),
        "avg_user_words": mean([m.get("avg_words_per_turn", 0) for m in metrics]) if metrics else 0,
        "avg_clarification_rate": mean([m.get("clarification_rate", 0) for m in metrics]) if metrics else 0,
        "avg_pushback_rate": mean([m.get("pushback_rate", 0) for m in metrics]) if metrics else 0,
        "early_termination_rate": mean([1.0 if m.get("early_termination") else 0.0 for m in metrics]) if metrics else 0,
        "persona_consistency_rate": mean([1.0 if c.get("consistent", True) else 0.0 for c in consistencies]) if consistencies else 1.0,
        "simulator_quality_score": mean([q.get("score", 1.0) for q in qualities]) if qualities else 1.0,
        "quality_issues": [issue for q in qualities for issue in q.get("issues", [])][:12],
    }


def analyze_failure_modes(results: dict, dialogues: list[dict]) -> dict:
    """从逐项判定中提取可落地的失败原因、证据和改进建议。"""
    failures = []
    unstable = []
    for did, result in results.items():
        item_votes = {}
        for run in result.get("run_details", []):
            for j in run.get("judgments", []):
                iid = j.get("item_id", "")
                item_votes.setdefault(iid, {"dialogue_id": did, "dimension": j.get("dimension"), "description": j.get("description", ""), "verdicts": [], "evidence": [], "reasons": []})
                item_votes[iid]["verdicts"].append(j.get("verdict"))
                if j.get("evidence") and j.get("evidence") != "未找到相关对话":
                    item_votes[iid]["evidence"].append(j.get("evidence"))
                if j.get("reason"):
                    item_votes[iid]["reasons"].append(j.get("reason"))
        for iid, info in item_votes.items():
            verdicts = info["verdicts"]
            if not verdicts:
                continue
            fail_rate = sum(1 for v in verdicts if v in ("NO", "PARTIAL")) / len(verdicts)
            if fail_rate > 0:
                failures.append({
                    "dialogue_id": info["dialogue_id"],
                    "item_id": iid,
                    "dimension": info["dimension"],
                    "description": info["description"],
                    "fail_rate": fail_rate,
                    "verdicts": verdicts,
                    "evidence": info["evidence"][:2],
                    "reason": info["reasons"][0] if info["reasons"] else "",
                })
            if len(set(verdicts)) > 1:
                unstable.append({"dialogue_id": info["dialogue_id"], "item_id": iid, "description": info["description"], "verdicts": verdicts})
    failures.sort(key=lambda x: (-x["fail_rate"], x["dimension"] or "", x["item_id"]))
    low_dimensions = []
    for did, result in results.items():
        for dim_key in DIMENSION_WEIGHTS:
            val = result.get(dim_key, {}).get("mean")
            if isinstance(val, (int, float)):
                low_dimensions.append((val, did, dim_key))
    low_dimensions.sort()
    return {
        "top_failures": failures[:12],
        "unstable_items": unstable[:8],
        "lowest_dimensions": low_dimensions[:8],
        "simulation_quality": summarize_simulation_quality(dialogues),
    }


def recommendation_for_dimension(dim_key: str) -> str:
    mapping = {
        "flow": "把指令流程编译成状态机，要求模型每轮只推进一步，并显式等待用户反馈。",
        "info": "将知识点拆成必说/条件触发/FAQ 三层，避免一次性信息倾倒。",
        "constraint": "把字数、禁用词、忙/开车等硬约束放入解码前检查或后处理拦截。",
        "opening": "开场白使用模板化变量填充，并用确定性覆盖率检查兜底。",
        "task": "增加用户理解确认环节，不只检查 Agent 说出信息，还检查用户是否接收。",
    }
    return mapping.get(dim_key, "补充针对该失败项的专门测试用例，并将其加入回归集。")


def run_full_evaluation(instruction: str, dialogues: list[dict], n_runs: int = 5, output_dir: str = None, report_name: str = "eval_v3_report.txt"):
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
    report = generate_report(instruction, results, cross_dialogue_stats, dialogues, output_dir=output_dir, report_name=report_name)
    
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


def generate_report(instruction: str, results: dict, cross_stats: dict, dialogues: list, output_dir: str = None, report_name: str = "eval_v3_report.txt") -> str:
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
    lines.append("【核心洞察】")
    lines.append("  复杂外呼指令失败通常不是单点知识遗漏，而是流程推进、用户打断、信息分步传达和硬约束共同作用的系统性失败。")
    lines.append("  本系统把自然语言指令先编译为冻结Rubric，再用多样化Persona做压力测试，最后输出可直接回流到模型/Prompt改进的失败项。")
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
    sim_quality = summarize_simulation_quality(dialogues)
    if sim_quality:
        lines.append("【用户模拟器质量】")
        lines.append(f"  平均用户回复长度: {sim_quality.get('avg_user_words', 0):.1f}字")
        lines.append(f"  追问率: {sim_quality.get('avg_clarification_rate', 0):.1%}")
        lines.append(f"  抵触率: {sim_quality.get('avg_pushback_rate', 0):.1%}")
        lines.append(f"  提前结束率: {sim_quality.get('early_termination_rate', 0):.1%}")
        lines.append(f"  Persona一致性: {sim_quality.get('persona_consistency_rate', 1):.1%}")
        lines.append(f"  模拟器质量分: {sim_quality.get('simulator_quality_score', 1):.3f}")
        for issue in sim_quality.get("quality_issues", [])[:5]:
            lines.append(f"  - 质量提示: {issue}")
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
    
    # 6. 失败诊断与改进建议
    failure_analysis = analyze_failure_modes(results, dialogues)
    if failure_analysis:
        lines.append("【失败诊断 Top Cases】")
        for f in failure_analysis.get("top_failures", [])[:8]:
            dim_name = DIMENSION_NAMES.get(f.get("dimension"), f.get("dimension"))
            lines.append(f"  - [{f.get('dialogue_id')}] {dim_name}/{f.get('item_id')} 失败率={f.get('fail_rate', 0):.0%}")
            lines.append(f"    检查项: {f.get('description', '')}")
            if f.get("evidence"):
                lines.append(f"    证据: {f['evidence'][0]}")
            if f.get("reason"):
                lines.append(f"    原因: {f.get('reason', '')[:160]}")
            lines.append(f"    建议: {recommendation_for_dimension(f.get('dimension'))}")
        lines.append("")
        if failure_analysis.get("unstable_items"):
            lines.append("【评测不稳定项】")
            for u in failure_analysis["unstable_items"][:5]:
                lines.append(f"  - [{u['dialogue_id']}] {u['item_id']}: {'/'.join(u['verdicts'])} — {u['description']}")
            lines.append("  这些项建议优先人工抽检，用于校准 Rubric 或补充判定示例。")
            lines.append("")
        if failure_analysis.get("lowest_dimensions"):
            lines.append("【瓶颈维度】")
            for val, did, dim_key in failure_analysis["lowest_dimensions"][:5]:
                lines.append(f"  - {did} / {DIMENSION_NAMES.get(dim_key, dim_key)} = {val:.3f}: {recommendation_for_dimension(dim_key)}")
            lines.append("")
    
    # 7. 方法论声明
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
    output_dir = output_dir or os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, report_name)
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
    
    parser = argparse.ArgumentParser(description="复杂指令下的多轮对话自动评测系统")
    parser.add_argument("--instruction-file", type=str, default=None,
                        help="任务指令文件，支持 .xlsx/.txt/.json；不传则使用内置示例")
    parser.add_argument("--instruction-id", type=str, default=None,
                        help="只评测指定任务ID，多个ID用逗号分隔")
    parser.add_argument("--dialogues", type=str, default=None,
                        help="复用已有对话数据文件(JSON)，通常只用于单任务调试")
    parser.add_argument("--n-runs", type=int, default=5,
                        help="每条对话评测次数 (默认5)")
    parser.add_argument("--n-dialogues", type=int, default=8,
                        help="每个任务生成的对话数 (默认8)")
    parser.add_argument("--skip-generate", action="store_true",
                        help="跳过对话生成，使用 outputs/simulated_dialogues_v2.json 或 v1 历史文件")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录，默认 outputs/hackathon_run")
    args = parser.parse_args()

    base_output_dir = args.output_dir or os.path.join(os.path.dirname(__file__), "..", "outputs", "hackathon_run")
    os.makedirs(base_output_dir, exist_ok=True)

    records = load_instruction_records(args.instruction_file, default_instruction=INSTRUCTION)
    records = filter_instruction_records(records, args.instruction_id)
    if not records:
        raise SystemExit("未找到可评测的任务指令")

    all_task_summaries = []
    for record in records:
        task_id = safe_slug(record.get("id", "task"))
        task_output_dir = os.path.join(base_output_dir, f"instruction_{task_id}")
        os.makedirs(task_output_dir, exist_ok=True)
        instruction = record.get("instruction", INSTRUCTION)

        print("\n" + "#" * 80)
        print(f"任务 {record.get('id')} | 来源: {record.get('source', 'unknown')}")
        print("#" * 80)

        if args.dialogues and os.path.exists(args.dialogues):
            with open(args.dialogues, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            instruction, dialogues = normalize_dialogue_payload(loaded, instruction)
            print(f"从文件加载 {len(dialogues)} 条对话: {args.dialogues}")
        elif args.skip_generate:
            v2_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "simulated_dialogues_v2.json")
            v1_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "simulated_dialogues.json")
            load_path = v2_path if os.path.exists(v2_path) else v1_path
            if os.path.exists(load_path):
                with open(load_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                instruction, dialogues = normalize_dialogue_payload(loaded, instruction)
                print(f"复用历史对话 {len(dialogues)} 条: {load_path}")
            else:
                print("未找到历史对话，将生成新对话")
                dialogues = generate_diverse_dialogues(instruction, n_dialogues=args.n_dialogues)
        else:
            print(f"生成 {args.n_dialogues} 条多样化对话...")
            dialogues = generate_diverse_dialogues(instruction, n_dialogues=args.n_dialogues)

        dialogue_bundle = {
            "instruction_id": record.get("id"),
            "source": record.get("source"),
            "instruction": instruction,
            "dialogues": [{
                "persona": d.get("persona", {}),
                "persona_type": d.get("persona_type", "unknown"),
                "dialogue": d.get("dialogue", []),
                "behavior_metrics": d.get("behavior_metrics", {}),
                "persona_consistency": d.get("persona_consistency", {}),
                "simulator_quality": d.get("simulator_quality", {}),
            } for d in dialogues],
        }
        dialogue_path = os.path.join(task_output_dir, "dialogues.json")
        with open(dialogue_path, "w", encoding="utf-8") as f:
            json.dump(dialogue_bundle, f, ensure_ascii=False, indent=2)
        print(f"对话数据已保存: {dialogue_path}")

        results, cross_stats, report = run_full_evaluation(
            instruction, dialogues, n_runs=args.n_runs,
            output_dir=task_output_dir,
            report_name="report.md",
        )

        save_results = {did: {k: v for k, v in r.items() if k not in ("run_details", "det_checks")} for did, r in results.items()}
        full_results_path = os.path.join(task_output_dir, "results.json")
        with open(full_results_path, "w", encoding="utf-8") as f:
            json.dump({
                "instruction_id": record.get("id"),
                "source": record.get("source"),
                "results": save_results,
                "cross_dialogue_stats": {k: v for k, v in cross_stats.items() if k != "ranking"},
                "failure_analysis": analyze_failure_modes(results, dialogues),
            }, f, ensure_ascii=False, indent=2)
        print(f"完整结果已保存: {full_results_path}")

        all_task_summaries.append({
            "instruction_id": record.get("id"),
            "source": record.get("source"),
            "n_dialogues": len(dialogues),
            "avg_score": mean([r.get("overall", {}).get("mean", 0) for r in results.values()]) if results else 0,
            "avg_stability": cross_stats.get("avg_stability"),
            "report": os.path.join(task_output_dir, "report.md"),
            "results": full_results_path,
        })

    index_path = os.path.join(base_output_dir, "summary.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({"tasks": all_task_summaries}, f, ensure_ascii=False, indent=2)
    print(f"\n批量评测完成，索引已保存: {index_path}")
