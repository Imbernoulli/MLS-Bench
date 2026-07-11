# Automatic Prompt Optimization: Shared PROPOSAL-vs-EVALUATION BUDGET ALLOCATION

## Research Question
A **frozen** small instruction LM (Qwen2.5-0.5B-Instruct) does **zero-shot** text classification — shown ONLY an instruction and the input, it predicts the label word it assigns the highest *contextually-calibrated* per-label likelihood (no demonstrations in the executed prompt). Automatic Prompt Engineering (APE) searches over the **instruction text**. Here ONE shared budget covers BOTH proposal (induction) and dev evaluation. Spending it all proposing many candidates leaves ~0 dev eval each (blind pick); balancing a few proposals with enough dev eval each picks the candidate that generalizes. The harness aborts on budget overrun. The chosen instruction is scored on a **held-out TEST set**.

This is a DISTINCT axis from in-context-learning (the icl-* tasks): there the demonstrations are selected/ordered/calibrated with the instruction FIXED; here the demonstrations are FIXED/ABSENT and the search is over the **INSTRUCTION TEXT**.

## Background
Zhou et al., "Large Language Models Are Human-Level Prompt Engineers" (ICLR 2023, [arXiv:2211.01910](https://arxiv.org/abs/2211.01910)); Honovich et al. 2022 ([arXiv:2205.10782](https://arxiv.org/abs/2205.10782)); Zhao et al. 2021, "Calibrate Before Use" ([arXiv:2102.09690](https://arxiv.org/abs/2102.09690)). The executor is **contextually calibrated** so instruction quality is monotone: an empty/degenerate instruction sits at the class prior; a genuine task description lifts accuracy. Because selection uses only DEV but the score is on the disjoint TEST set, a dev-overfit choice does not win.

Reference baselines (read-only edit ops):
- **allpropose** — weak negative control (degenerate allocate).
- **balanced** — strong reference for this axis.

## What is FIXED (you cannot change)
The base LM, the calibrated zero-shot forced-choice executor, the DEV set, the HELD-OUT TEST set, the label set, and the selection rule. You control **only** the `allocate(ctx) -> str` proposal/eval budget allocator.

## Model Interface
Implement `allocate` in `prompt-optimization-lab/solution/budget.py`:
```python
def allocate(ctx)::
    # see solution/budget.py for the full signature and ctx tools
    ...
```

## Datasets & Metric
- **agnews** — AG News, 4-class topic (majority floor 0.25).
- **sst2** — SST-2, 2-class sentiment (majority floor 0.50; official evaluation split).
Metric = zero-shot **accuracy** on the held-out TEST set of the instruction your surface selects, higher is better. Inference-only, single GPU, a few minutes.

## Evaluation Protocol
- **agnews** - all 7,600 rows of the official AG News test split.
- **sst2** - all 872 labeled rows of the official SST-2 validation split.

The proposal pool (128 rows) and candidate-selection set (200 rows) are drawn only
from the official training split. Their texts are disjoint from each other and from
the evaluation split. Candidate generation and ranking use only train/dev data; the
verifier runs exactly one selected instruction on the complete evaluation split.
Both dataset settings participate in the score. The settings run serially on one GPU.
