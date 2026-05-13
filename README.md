# agent-eval

外呼任务对话 Agent 的评测系统：**LLM 模拟用户 → Agent-as-a-Judge 打分**，配合统计严谨性指标（BCa CI / ICC / Gwet's AC1 / Krippendorff's α / pass^k）输出可解释、可量化的评测报告。

> 项目动机见 [`INTENT.md`](./INTENT.md)，论文调研见 [`SURVEY.md`](./SURVEY.md)。这两篇是阅读代码前的"为什么"。

## 系统架构

```
data/instruction_*.txt   ──►  user_simulator_v2.py  ──►  outputs/*.json (dialogues)
                                          │
                                          ▼
                                  eval_v3.py (Agent-as-a-Judge)
                                  ├─ Phase I:   compile_rubric()        原子化检查清单
                                  ├─ Phase II:  judge_single_item()     逐项判定 + 证据锚定
                                  ├─           deterministic_checks()  代码验证（字数/禁用词）
                                  └─ Phase III: compute_scores()        加权 + 跨维度校准
```

`eval_v3_runner.py` 把整套流水线串起来：模拟对话 → 多次评测（默认 n=5）→ 跨对话统计 → 报告。

代码经历过三代演进：

| 代际 | 入口 | 状态 |
|------|------|------|
| v1 | `run_eval.py` + `eval_agent.py` | 基线（OpenAI SDK）|
| v2 | `run_e2e.py` + `strict_eval.py` + `user_simulator.py` | 加入统计严谨性 |
| **v3** | **`eval_v3_runner.py` + `eval_v3.py` + `user_simulator_v2.py`** | **当前主线** |

## 快速开始

### 1. 配置 LLM 代理

```bash
cp .env.example .env
# 编辑 .env，填入 ADAMS_* 三个 token
source .env
```

> 项目通过裸 socket 直连内部代理 `mmdcadamsminiserverproxy.polaris:25340`（绕过 HTTP 客户端限制），调用 `kimi_k2d6` 模型。无 token 时所有 LLM 请求都会被拒。

### 2. 大赛交付入口：读取官方 Excel 批量评测

```bash
# 跑官方 Excel 中全部任务：每个任务生成 8 条多样化对话，每条评测 5 次
python src/eval_v3_runner.py \
  --instruction-file "命题二：外呼任务对话模型指令示例 (1).xlsx" \
  --n-dialogues 8 \
  --n-runs 5

# 快速演示：只跑第 2 条任务，减少对话数和评测次数
python src/eval_v3_runner.py \
  --instruction-file data/instructions.json \
  --instruction-id 2 \
  --n-dialogues 2 \
  --n-runs 2
```

输出目录默认在 `outputs/hackathon_run/`，每个任务独立保存：

- `dialogues.json`：模拟用户、多轮对话、persona 与 simulator quality。
- `report.md`：可解释评测报告、失败诊断、证据和改进建议。
- `results.json`：机器可读分数、可靠性指标、失败分析。
- `summary.json`：批量任务索引。

### 3. 调试入口

```bash
# 复用已有对话（跳过生成）
python src/eval_v3_runner.py --skip-generate --n-runs 5

# 评测指定对话文件
python src/eval_v3_runner.py --dialogues outputs/simulated_dialogues_v2.json --n-runs 3
```

### 3. 其他入口

```bash
python src/eval_v3.py        # v3 自检（内置 good vs bad 对话，n_runs=2）
python src/run_e2e.py        # v2 端到端：5 种固定 persona
python src/run_eval.py       # v1 demo
python src/run_eval_quick.py # thinking vs non-thinking 对比
python src/human_annotate.py # 交互式人工标注
```

## 评测方法（v3 关键设计）

1. **Rubric 一次编译，跨对话冻结复用**（`compile_rubric()` + hash），保证同一指令下所有对话评测标准一致。
2. **5 维度加权打分**：流程 (2.0) / 信息 (2.0) / 约束 (1.5) / 开场白 (1.5) / 任务 (1.5)；每维度最多 8 项。
3. **YES / PARTIAL / NO / N/A** 四档，对应 1.0 / 0.5 / 0.0 / null。N/A 不计入分母。
4. **证据锚定**：每个 YES 必须附带对话原文，`verify_evidence()` 找不到则降级为 PARTIAL（不归零）。
5. **确定性校准**：字数、禁用词等用代码直接检查，作为约束维度的上界，**不与 LLM 判定混合**。
6. **跨维度校准**：信息在错误步骤传达 → 打折；约束分极低 → info/task 跟着打折。
7. **用户模拟器自检**：检测元评论泄漏、过长回复、persona 不一致、过度配合等 sim-to-real 风险。
8. **失败诊断报告**：从 `run_details` 汇总 Top 失败项、证据、原因、瓶颈维度和可执行改进建议。
9. **多次运行**：n_runs ≥ 2，报告 BCa 95% CI、ICC(1)、Gwet's AC1、Krippendorff's α、pass^k。

灵感来源（详见 SURVEY.md）：Agent-as-a-Judge (Zhuge 2024)、RULERS (Hong 2026)、Rubric Is All You Need (Pathak 2025)、τ-bench (Yao 2024)、Sim2Real Gap (Zhou 2026)、Persona-driven (Gromada EMNLP 2025)。

## 依赖

无 `requirements.txt`。除 `eval_agent.py` 用到 `openai` 包外，其余代码全部基于 Python 标准库——统计函数（BCa bootstrap / Gwet AC1 / Krippendorff α / 正态 CDF/PPF）都是手写的，刻意避免 `numpy`/`scipy` 以保持零依赖。

```bash
pip install openai  # 仅 v1 路径需要
```

## 目录结构

```
agent-eval/
├── INTENT.md                # 产品意图
├── SURVEY.md                # 论文调研
├── CLAUDE.md                # 给 Claude Code 的开发指南
├── data/                    # 输入：任务指令
│   ├── instructions.json    # 从官方 Excel 标准化抽取的任务指令
│   ├── instruction_1.txt    # 站长外呼骑手（飞毛腿）
│   └── instruction_2.txt    # 客服外呼机构（直播升级）
├── src/                     # 三代评测器代码
│   └── instruction_loader.py # 零依赖读取 xlsx/txt/json 指令
└── outputs/                 # 生成的对话 + 评测结果
```

## 注意

- 代码注释、prompt、维度名都是中文。修改 prompt 时务必保留 JSON / `<think>/<verdict>/<score>/<evidence>` 输出格式（解析器是严格匹配的）。
- 7 个文件各自维护一份 `call_llm()` 副本，这是有意的——内部代理只对裸 socket 友好，**不要**重构成 `requests`/`httpx`。
- `eval_v3_clean.py` 是 `eval_v3.py` 的近重复版本，bug 修复请改 `eval_v3.py`。
