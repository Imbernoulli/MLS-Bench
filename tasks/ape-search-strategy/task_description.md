# Automatic Prompt Optimization: Budgeted Search Strategy

## Research Question
A **frozen** small instruction LM (Qwen2.5-0.5B-Instruct) does **zero-shot** text
classification — shown ONLY an instruction and the input, it predicts the label word
with the highest *contextually-calibrated* per-label likelihood (no demonstrations in
the executed prompt). Automatic Prompt Engineering (APE) ranks candidate instructions
by their DEV execution-accuracy, but evaluating every candidate on the whole dev set
is expensive, so a strict **budget** caps the number of (candidate, dev-example)
executions. The candidate POOL and the estimator (dev execution-accuracy) are FIXED —
you design ONLY the **search/allocation** `select(candidates, ctx)`: how many dev
examples to spend on which candidates. The chosen instruction is scored on a
**held-out TEST set**.

This is a DISTINCT axis from in-context-learning (the icl-* tasks): there the
demonstrations are selected/ordered/calibrated with the instruction FIXED; here the
demonstrations are FIXED/ABSENT and the search is over the **INSTRUCTION TEXT**.

## Background
Choosing the best of many candidate instructions under a limited evaluation budget is
a best-arm-identification problem. Judging every candidate on a TINY dev slice gives
high-variance estimates that overfit dev noise and often pick a candidate that does
not generalize; **successive halving** (Jamieson & Talwalkar 2016,
[arXiv:1502.07943](https://arxiv.org/abs/1502.07943)) — cheaply screen all candidates,
then re-evaluate the survivors on MORE dev examples — gives the finalists a stable
estimate under the SAME budget, so the chosen candidate transfers to the disjoint
held-out TEST set. Zhou et al. 2022 ([arXiv:2211.01910](https://arxiv.org/abs/2211.01910))
likewise use a filtered/iterative dev-scoring scheme to spend an APE search budget on
the most promising candidates. The forced-choice executor is contextually calibrated
(Zhao et al. 2021, "Calibrate Before Use") so dev accuracy is a meaningful, monotone
signal.

Reference baselines (provided as read-only edit ops):
- **first** — pick candidates[0]; no search, no budget spent (poor generalization).
- **tiny** — judge every candidate on the same tiny dev slice (high variance).
- **halving** — successive halving: screen all, re-evaluate survivors on more dev.

## What is FIXED (you cannot change)
The base LM, the calibrated zero-shot forced-choice executor, the candidate POOL, the
dev-accuracy estimator, the DEV/TEST split, the label set, and the execution BUDGET.
The executor COUNTS executions and the harness ABORTS you if you exceed the budget, so
"evaluate everything on all of dev" is impossible. You control **only** the allocation
in `select`.

## Model Interface
Implement `select` in `prompt-optimization-lab/solution/strategy.py`:
```python
def select(candidates, ctx) -> str:
    # candidates : list[str] the FIXED candidate instructions
    # ctx["dev"]    a LARGE dev set (sub-sample it!)
    # ctx["budget"] max (candidate, dev-example) executions allowed
    # ctx["eval_on_dev"](instruction, dev_rows) -> float
    #     budget-guarded dev accuracy; the harness ABORTS you if you exceed budget
    # ctx["rng"], ctx["dataset"]
    # return ONE instruction string from `candidates`.
    ...
```

## Datasets & Metric
- **agnews** — AG News, 4-class topic classification (majority floor 0.25).
- **sst2** — SST-2, 2-class sentiment (majority floor 0.50; official evaluation split).
Metric = zero-shot classification **accuracy** on the held-out TEST set of the
candidate your search selects, higher is better. Because the budget is small relative
to (#candidates × #dev) and selection uses only DEV, a naive/degenerate allocation
picks a poorly-generalizing candidate; a variance-aware allocation generalizes to
TEST. Inference-only, single GPU, a few minutes.

## Evaluation Protocol
- **agnews** - all 7,600 rows of the official AG News test split.
- **sst2** - all 872 labeled rows of the official SST-2 validation split.

The proposal pool (128 rows) and candidate-selection set (200 rows) are drawn only
from the official training split. Their texts are disjoint from each other and from
the evaluation split. Candidate generation and ranking use only train/dev data; the
verifier runs exactly one selected instruction on the complete evaluation split.
Both dataset settings participate in the score. The settings run serially on one GPU.
