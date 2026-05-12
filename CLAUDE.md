# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

This is a research-prototype evaluation system for outbound-call dialogue agents (外呼任务对话模型). The product thesis is in `INTENT.md` (Chinese): the project bets on the "simulation school" of agent evaluation — LLM-simulated users + engineering constraints to close the sim-to-real gap. `SURVEY.md` catalogs the papers each component is inspired by (τ-bench, Agent-as-a-Judge, RULERS, "Rubric Is All You Need", the Sim2Real Gap paper, Persona-driven simulation, etc.). Read both when changing the architecture — the design choices in code are deliberate and trace back to specific findings in those papers.

There is **no build system, no test runner, no dependency manifest**. Scripts are run directly with `python src/<file>.py`. Files in `outputs/` are generated artifacts (committed for inspection); files in `data/` are the input task instructions.

## How the pieces fit together

The system has evolved through three generations. Treat **v3 as the current line** — older modules are kept for comparison/regression but are not the path forward.

```
data/instruction_*.txt   ──► user_simulator_v2.py  ──► outputs/*.json (dialogues)
                                       │
                                       ▼
                              eval_v3.py  (Agent-as-a-Judge)
                              ├─ Phase I:  compile_rubric()       → atomic checklist
                              ├─ Phase II: judge_single_item()    → per-item verdict + evidence
                              ├─       deterministic_checks()     → code-verified upper bound
                              └─ Phase III: compute_scores()      → weighted, calibrated score

eval_v3_runner.py orchestrates: simulate → evaluate (n_runs each) → cross-dialogue stats → report
```

**Two parallel pipelines exist** (this is intentional, do not collapse them without reason):
- `eval_agent.py` + `run_eval.py` — the original v1 path using `openai` SDK and `dataclass`-based `EvaluationAgent`. Kept as a baseline.
- `strict_eval.py` + `run_e2e.py` — v2 path with statistical rigor (ICC, bootstrap CI), uses `user_simulator.py` (v1 personas).
- `eval_v3.py` + `eval_v3_runner.py` — current path. Adds rubric compilation, evidence anchoring, GRM-style judging, BCa bootstrap, Gwet's AC1, Krippendorff's α, pass^k. Uses `user_simulator_v2.py` (parameterized personas).

`eval_v3_clean.py` is a near-duplicate of `eval_v3.py` (kept for diff/cleanup). When fixing bugs, fix `eval_v3.py` — `eval_v3_clean.py` is downstream.

`parse_grm.py` is extracted from `eval_v3.py` and exists because the v3 LLM output uses `<think>/<verdict>/<score>/<evidence>` tags (GRM = Generative Reward Model) rather than JSON. Both formats are accepted as a fallback chain.

## The LLM call (important constraint)

All four "main" modules (`eval_v3.py`, `strict_eval.py`, `user_simulator_v2.py`, `run_eval_quick.py`) each carry their **own copy** of `call_llm()`. They all do the same thing: hit `mmdcadamsminiserverproxy.polaris:25340` over **raw sockets**, bypassing any HTTP client / proxy. The model is `kimi_k2d6`. Auth headers (`Adams-Platform-User`, `Adams-User-Token`, `Adams-Business`) are inlined.

Why raw sockets: the proxy environment blocks normal `requests`/`httpx`. Do **not** "refactor" these into `openai.OpenAI` — `eval_agent.py` does use the OpenAI SDK and only works because it's pointed at the same proxy via `base_url`. Touching this transport breaks every module.

If you need to change LLM behavior (timeout, IP cache, chunked decoding), `eval_v3.py:53-135` has the most evolved version (multi-IP failover with `_last_good_ip` cache). Mirror changes to the other copies.

## Evaluation methodology — what makes a verdict valid

The v3 scoring contract is non-obvious and must be preserved when editing:

1. **Rubric is compiled per-instruction, then frozen** (`compile_rubric()` + `hash` field). Reuse the same rubric across dialogues for the same instruction; do not recompile per dialogue (see `eval_v3_runner.py:54-58`).
2. **Five dimensions with fixed weights** (`DIMENSION_WEIGHTS` in `eval_v3.py:825`): `flow=2.0, info=2.0, constraint=1.5, opening=1.5, task=1.5`. Items per dimension are capped at `MAX_PER_DIM=8`.
3. **Verdicts are YES / PARTIAL / NO / N/A**, mapping to scores `1.0 / 0.5 / 0.0 / null`. N/A means the precondition didn't trigger and **must not** count toward the denominator.
4. **Evidence anchoring**: every YES requires a verbatim quote that `verify_evidence()` can find in the dialogue. Failed verification **demotes YES to PARTIAL** (it does not zero the score). See `eval_v3.py:644-657`.
5. **Deterministic checks (`deterministic_checks()`) are not LLM checks** — they are the upper bound for the `constraint` dimension. Word-count and banned-word violations are computed in pure Python and reported alongside, never merged into, LLM verdicts.
6. **Cross-dimension calibration**: `info ≤ flow` (info delivered in the wrong step is discounted), and very low `constraint` discounts `info`/`task` (information dump = information not received). This is in `compute_scores()`.
7. **Multiple runs are mandatory**: `n_runs ≥ 2`. Reports always include BCa 95% CI, ICC(1), Gwet's AC1, Krippendorff's α, and pass^k stability — never single-run scores.

## Common commands

Everything runs from the repo root with `python src/<file>.py`. The scripts read/write `outputs/` relative to `src/__file__`, so the working directory mostly does not matter, but the repo layout (`src/`, `outputs/`, `data/` as siblings) does.

```bash
# Full v3 pipeline: generate dialogues + evaluate + cross-dialogue stats
python src/eval_v3_runner.py --n-dialogues 8 --n-runs 5

# Reuse previously simulated dialogues (skip generation)
python src/eval_v3_runner.py --skip-generate --n-runs 5

# Evaluate dialogues from a specific file
python src/eval_v3_runner.py --dialogues outputs/simulated_dialogues_v2.json --n-runs 3

# v3 self-test (built-in good vs bad dialogue, n_runs=2 hardcoded in main)
python src/eval_v3.py

# v2 end-to-end (5 fixed personas: cooperative/reluctant/rejecting/curious/driving)
python src/run_e2e.py

# v1 demo with hardcoded good/bad dialogues
python src/run_eval.py

# Quick eval comparing thinking vs non-thinking modes
python src/run_eval_quick.py

# Human-in-the-loop annotation (interactive prompts)
python src/human_annotate.py
```

There is no `pytest` or test directory. The "tests" are the `main()` functions in `eval_v3.py` / `run_eval.py` / `run_e2e.py` — they run a fixed good-vs-bad dialogue pair and assert (informally, by printing) that good > bad. When debugging a scoring change, run `python src/eval_v3.py` first; it's the fastest sanity check.

## Working-with-this-code notes

- **No external deps file.** Scripts assume `openai` is importable (used only by `eval_agent.py`); everything else uses stdlib (`socket`, `json`, `re`, `random`, `math`). Don't add `numpy`/`scipy` — the statistical code (`bca_bootstrap_ci`, `gwet_ac1`, `krippendorffs_alpha`, `_normal_ppf`, `_normal_cdf`) is hand-rolled deliberately to keep the module zero-dep.
- **All prompts are Chinese** and the dimension names are Chinese. When editing prompts, preserve the JSON/XML output schema — parsers in `parse_grm.py` and `parse_json_from_llm` are strict.
- **Auth via env vars.** Each module reads `ADAMS_PLATFORM_USER`, `ADAMS_USER_TOKEN`, `ADAMS_BUSINESS` from the environment. See `.env.example` — copy to `.env` and `source .env` (or export inline) before running. Empty values mean every LLM call will be rejected by the proxy.
- The Excel file in repo root (`命题二：...xlsx`) is the original task brief. `data/instruction_*.txt` are extracted task definitions derived from it.
- When adding a new instruction, drop a `data/instruction_N.txt` and pass it as the `instruction` arg — `compile_rubric()` will produce a fresh rubric. Do not hardcode dimension counts; they're driven by the rubric.
