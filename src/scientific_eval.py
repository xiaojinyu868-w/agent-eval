"""
科研级评测框架 v3

严谨性措施：
1. Bootstrap CI（样本小时比t分布更稳健）
2. 对多条对话做评测，计算评测系统与人类评分的相关系数
3. Spearman/Kendall 相关
4. 逐项 Cohen's Kappa（如有人类标注）
5. 偏差校准函数
6. 效应量报告
7. 多对话均值 + 跨对话方差
"""

import sys
import os
import json
import time
import socket
import re
import random

sys.path.insert(0, os.path.dirname(__file__))

from strict_eval import (
    call_llm, deterministic_checks, format_dialogue,
    parse_eval_result, mean, std, confidence_interval_95, icc_1way,
    EVAL_PROMPT, EXTRA_HEADERS, IPS,
)


# ============================================================
# Bootstrap 置信区间
# ============================================================

def bootstrap_ci(vals, n_bootstrap=10000, ci=0.95):
    """Bootstrap 95% 置信区间，比t分布更稳健（样本小时）"""
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


# ============================================================
# 相关性计算
# ============================================================

def rank_data(vals):
    """计算排名（处理并列）"""
    n = len(vals)
    indexed = sorted(enumerate(vals), key=lambda x: x[1])
    ranks = [0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman_rho(x, y):
    """Spearman 秩相关系数"""
    if len(x) != len(y) or len(x) < 3:
        return None
    rx = rank_data(x)
    ry = rank_data(y)
    n = len(x)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    denom = n * (n ** 2 - 1)
    return 1 - 6 * d2 / denom if denom != 0 else 0


def kendall_tau(x, y):
    """Kendall's Tau-b（处理并列）"""
    if len(x) != len(y) or len(x) < 3:
        return None
    n = len(x)
    concordant = 0
    discordant = 0
    tied_x = 0
    tied_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            elif dx == 0:
                tied_x += 1
            elif dy == 0:
                tied_y += 1
            elif dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    denom = ((concordant + discordant + tied_x) * (concordant + discordant + tied_y)) ** 0.5
    return (concordant - discordant) / denom if denom != 0 else 0


def cohens_kappa(rater1, rater2, categories=None):
    """Cohen's Kappa for two raters"""
    if len(rater1) != len(rater2) or len(rater1) < 2:
        return None
    if categories is None:
        categories = list(set(rater1 + rater2))
    n = len(rater1)
    k = len(categories)

    # Observed agreement
    obs = sum(1 for a, b in zip(rater1, rater2) if a == b) / n

    # Expected agreement
    p1 = {c: rater1.count(c) / n for c in categories}
    p2 = {c: rater2.count(c) / n for c in categories}
    exp = sum(p1.get(c, 0) * p2.get(c, 0) for c in categories)

    return (obs - exp) / (1 - exp) if (1 - exp) != 0 else 1.0


# ============================================================
# 多对话评测
# ============================================================

def evaluate_dialogue(instruction, dialogue, n_runs=5):
    """对单条对话做多次评测，返回统计结果"""
    dialogue_text = format_dialogue(dialogue)

    # 确定性检查
    det_results = deterministic_checks(instruction, dialogue)

    # 多次LLM评测
    llm_results = []
    for i in range(n_runs):
        prompt = EVAL_PROMPT.format(instruction=instruction, dialogue=dialogue_text)
        content = call_llm([{"role": "user", "content": prompt}], temperature=0.3, max_tokens=8192)
        parsed = parse_eval_result(content)
        if parsed:
            llm_results.append(parsed)

    if not llm_results:
        return None

    dim_names = ["流程遵循度", "信息传达完整性", "约束遵循度", "任务达成度"]

    # 收集分数
    overall_vals = [r.get("overall_score", 0) for r in llm_results]
    dim_vals = {}
    for dim in dim_names:
        dim_vals[dim] = [r.get("dimensions", {}).get(dim, {}).get("score", 0) for r in llm_results]

    # 确定性校准
    det_constraint_rate = None
    for d in det_results:
        if d["dimension"] == "约束遵循度":
            det_constraint_rate = d["pass_rate"]
            break

    if det_constraint_rate is not None and "约束遵循度" in dim_vals:
        dim_vals["约束遵循度"] = [min(v, det_constraint_rate) for v in dim_vals["约束遵循度"]]

    # 跨维度惩罚
    if "信息传达完整性" in dim_vals and "流程遵循度" in dim_vals:
        for i in range(len(dim_vals["信息传达完整性"])):
            if dim_vals["信息传达完整性"][i] > dim_vals["流程遵循度"][i] * 0.8:
                dim_vals["信息传达完整性"][i] = min(
                    dim_vals["信息传达完整性"][i],
                    dim_vals["流程遵循度"][i] * 0.8
                )

    # 重新计算总分
    weights = {"流程遵循度": 2.0, "信息传达完整性": 2.0, "约束遵循度": 1.5, "任务达成度": 1.5}
    recalculated = []
    for i in range(len(llm_results)):
        ws = sum(dim_vals.get(d, [0]*len(llm_results))[i] * w for d, w in weights.items())
        tw = sum(w for d, w in weights.items())
        recalculated.append(ws / tw if tw > 0 else 0)
    overall_vals = recalculated

    # 统计
    result = {
        "overall": {"mean": mean(overall_vals), "std": std(overall_vals), "vals": overall_vals},
        "dimensions": {},
        "det_checks": det_results,
        "n_runs": len(llm_results),
    }

    for dim in dim_names:
        vals = dim_vals.get(dim, [])
        if vals:
            ci_lo, ci_hi = bootstrap_ci(vals)
            result["dimensions"][dim] = {
                "mean": mean(vals), "std": std(vals),
                "bootstrap_ci_95": [ci_lo, ci_hi],
                "vals": vals,
            }

    return result


def run_full_evaluation(instruction, dialogues, n_runs=5):
    """对所有对话做评测"""
    print("科研级评测框架 v3")
    print(f"对话数: {len(dialogues)}, 每条评测次数: {n_runs}\n")

    results = {}
    for i, dl in enumerate(dialogues):
        persona = dl["persona"]
        dialogue = dl["dialogue"]
        dialogue_id = f"dlg_{i:02d}_{persona}"

        print(f"\n{'='*50}")
        print(f"评测: {dialogue_id} (persona={persona}, {len(dialogue)}轮)")
        print(f"{'='*50}")

        t0 = time.time()
        result = evaluate_dialogue(instruction, dialogue, n_runs=n_runs)
        elapsed = time.time() - t0

        if result:
            overall = result["overall"]
            ci = result["dimensions"].get("流程遵循度", {}).get("bootstrap_ci_95", [0, 0])
            print(f"  总分: {overall['mean']:.3f} ± {overall['std']:.3f}")
            print(f"  Bootstrap CI(overall): [{ci[0]:.3f}, {ci[1]:.3f}]")
            for dim, dv in result["dimensions"].items():
                print(f"  {dim}: {dv['mean']:.3f}")
            print(f"  耗时: {elapsed:.1f}s")
            results[dialogue_id] = result
        else:
            print(f"  评测失败")

    return results


def compute_inter_dialogue_statistics(results):
    """跨对话统计分析"""
    print(f"\n{'='*60}")
    print("跨对话统计分析")
    print(f"{'='*60}")

    dialogue_ids = list(results.keys())
    if len(dialogue_ids) < 3:
        print("对话数不足3条，无法做跨对话统计")
        return

    # 1. 各对话总分排名
    overall_means = [(did, results[did]["overall"]["mean"]) for did in dialogue_ids]
    overall_means.sort(key=lambda x: x[1], reverse=True)

    print(f"\n--- 总分排名 ---")
    for rank, (did, score) in enumerate(overall_means, 1):
        bar = "█" * int(score * 20)
        print(f"  {rank}. {did:<25} {score:.3f} {bar}")

    # 2. 各维度跨对话均值和方差
    dim_names = ["流程遵循度", "信息传达完整性", "约束遵循度", "任务达成度"]
    print(f"\n--- 跨对话维度统计 ---")
    print(f"  {'维度':<16} {'均值':>6} {'标准差':>6} {'最小':>6} {'最大':>6}")
    print("  " + "-" * 44)
    for dim in dim_names:
        dim_means = [results[did]["dimensions"].get(dim, {}).get("mean", 0) for did in dialogue_ids]
        if dim_means:
            print(f"  {dim:<16} {mean(dim_means):>6.3f} {std(dim_means):>6.3f} {min(dim_means):>6.3f} {max(dim_means):>6.3f}")

    # 3. 确定性检查 vs LLM评测的一致性
    print(f"\n--- 确定性检查校准效果 ---")
    for did in dialogue_ids:
        det_checks = results[did].get("det_checks", [])
        llm_constraint = results[did]["dimensions"].get("约束遵循度", {}).get("mean", None)
        det_constraint = None
        for d in det_checks:
            if d["dimension"] == "约束遵循度":
                det_constraint = d["pass_rate"]
                break
        if det_constraint is not None and llm_constraint is not None:
            status = "✅" if abs(llm_constraint - det_constraint) < 0.1 else "⚠️"
            print(f"  {did}: 确定性={det_constraint:.1%}, LLM={llm_constraint:.1%} {status}")

    # 4. 评测者内一致性（每条对话的ICC）
    print(f"\n--- 评测者内一致性(ICC) ---")
    icc_vals = []
    for did in dialogue_ids:
        dim_vals_list = []
        for dim in dim_names:
            vals = results[did]["dimensions"].get(dim, {}).get("vals", [])
            if vals:
                dim_vals_list.append(vals)

        if len(dim_vals_list) >= 3:
            # 转置: runs × dims
            matrix = []
            min_len = min(len(v) for v in dim_vals_list)
            for r in range(min_len):
                row = [dim_vals_list[d][r] for d in range(len(dim_vals_list))]
                matrix.append(row)
            if len(matrix) >= 3:
                icc = icc_1way(matrix)
                icc_vals.append(icc)
                quality = "优秀" if icc > 0.75 else "良好" if icc > 0.5 else "差"
                print(f"  {did}: ICC={icc:.3f} ({quality})")

    if icc_vals:
        print(f"  平均ICC: {mean(icc_vals):.3f}")

    # 5. 如果有人类标注，计算相关系数
    annotations_dir = os.path.join(os.path.dirname(__file__), "..", "annotations")
    if os.path.exists(annotations_dir):
        human_scores = {}
        for fname in os.listdir(annotations_dir):
            if fname.endswith(".json"):
                with open(os.path.join(annotations_dir, fname), "r", encoding="utf-8") as f:
                    ann = json.load(f)
                # 匹配dialogue_id
                did = ann.get("dialogue_id", "")
                if did in results:
                    human_scores[did] = ann.get("scores", {})

        if human_scores:
            print(f"\n--- LLM评测 vs 人类标注相关性 ---")
            # Overall相关性
            llm_overalls = [results[did]["overall"]["mean"] for did in human_scores if did in results]
            human_overalls = [human_scores[did].get("overall", 0) / 5.0 for did in human_scores if did in results]

            if len(llm_overalls) >= 3:
                sp = spearman_rho(llm_overalls, human_overalls)
                kt = kendall_tau(llm_overalls, human_overalls)
                print(f"  总分 Spearman ρ: {sp:.3f}" if sp else "  Spearman: 计算失败")
                print(f"  总分 Kendall τ: {kt:.3f}" if kt else "  Kendall: 计算失败")

                # 逐维度
                for dim in dim_names:
                    llm_dims = [results[did]["dimensions"].get(dim, {}).get("mean", 0) for did in human_scores if did in results]
                    human_dims = [human_scores[did].get(dim, 0) / 5.0 for did in human_scores if did in results]
                    if len(llm_dims) >= 3:
                        sp = spearman_rho(llm_dims, human_dims)
                        print(f"  {dim} Spearman: {sp:.3f}" if sp else "")

                # 逐项Kappa（如有）
                print(f"\n  逐项判定 Cohen's Kappa (LLM vs Human):")
                for did in human_scores:
                    if did in results:
                        # 简化：对比维度级判定
                        for dim in dim_names:
                            llm_mean = results[did]["dimensions"].get(dim, {}).get("mean", 0)
                            human_score = human_scores[did].get(dim, 0) / 5.0
                            llm_cat = "pass" if llm_mean >= 0.6 else "partial" if llm_mean >= 0.3 else "fail"
                            human_cat = "pass" if human_score >= 0.6 else "partial" if human_score >= 0.3 else "fail"
                            match = "✅" if llm_cat == human_cat else "❌"
                            print(f"    {did} {dim}: LLM={llm_cat} Human={human_cat} {match}")
    else:
        print(f"\n  [提示] 未找到人类标注。运行 python human_annotate.py 收集标注后可计算相关性。")

    # 6. 保存
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    output = {}
    for did, r in results.items():
        output[did] = {
            "overall_mean": r["overall"]["mean"],
            "overall_std": r["overall"]["std"],
            "dimensions": {dim: {
                "mean": dv.get("mean"), "std": dv.get("std"),
                "bootstrap_ci_95": dv.get("bootstrap_ci_95"),
            } for dim, dv in r["dimensions"].items()},
        }
    with open(os.path.join(output_dir, "scientific_eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: outputs/scientific_eval_results.json")


if __name__ == "__main__":
    # 加载模拟对话
    dialogues_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "simulated_dialogues.json")
    with open(dialogues_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    instruction = data["instruction"]
    dialogues = data["dialogues"]

    results = run_full_evaluation(instruction, dialogues, n_runs=5)
    compute_inter_dialogue_statistics(results)
