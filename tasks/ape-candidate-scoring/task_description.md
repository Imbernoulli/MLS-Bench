# Automatic Prompt Optimization: Candidate Scoring

## Research Question
A **frozen** small instruction LM (Qwen2.5-0.5B-Instruct) does **zero-shot** text
classification — shown ONLY an instruction and the input, it predicts the label word
it assigns the highest *contextually-calibrated* per-label likelihood (no
demonstrations in the executed prompt). Automatic Prompt Engineering (APE) proposes
many candidate instructions and must **pick the one that generalizes**. Here the
candidate POOL and the search rule are FIXED — the harness hands you a fixed list of
candidate instructions and, after you score them, picks the single highest-scoring
one and evaluates it on a **held-out TEST set**. You design ONLY the **estimator**
`score_candidate(instruction, ctx)` that ranks candidates on the DEV set.

This is a DISTINCT axis from in-context-learning (the icl-* tasks): there the
demonstrations are selected/ordered/calibrated with the instruction FIXED; here the
demonstrations are FIXED/ABSENT and the search is over the **INSTRUCTION TEXT**.

## Background
Zhou et al., "Large Language Models Are Human-Level Prompt Engineers" (ICLR 2023,
[arXiv:2211.01910](https://arxiv.org/abs/2211.01910)) score each candidate
instruction by its **execution accuracy** on a dev set and keep the best; Honovich et
al. 2022 ([arXiv:2205.10782](https://arxiv.org/abs/2205.10782)) study instruction
induction and estimators. The estimator is the crux: a good one surfaces the
candidate the small LM actually follows and that transfers to unseen inputs; a
random/constant estimator selects arbitrarily and, with the misleading distractor
instructions in the pool, lands near the class prior. The forced-choice executor is
**contextually calibrated** (Zhao et al. 2021, "Calibrate Before Use",
[arXiv:2102.09690](https://arxiv.org/abs/2102.09690)) so instruction quality is
monotone: an empty/degenerate instruction sits at the class prior, a genuine task
description lifts accuracy.

Reference baselines (provided as read-only edit ops):
- **random** — constant score; the argmax picks an arbitrary (often misleading)
  candidate (near class prior).
- **execacc** — dev execution-accuracy of each candidate (canonical APE metric).
- **loglik** — mean gold-label calibrated log-likelihood on dev (a smoother signal).

## What is FIXED (you cannot change)
The base LM, the calibrated zero-shot forced-choice executor, the candidate POOL, the
greedy argmax search rule, the DEV set, the HELD-OUT TEST set, and the label set. You
control **only** the scalar rank-score `score_candidate` returns per candidate.

## Model Interface
Implement `score_candidate` in `prompt-optimization-lab/solution/scoring.py`:
```python
def score_candidate(instruction, ctx) -> float:
    # ctx["executor"].dev_accuracy(instruction, dev_rows) -> float
    # ctx["executor"].label_logprob_matrix(instruction, rows) -> [n,n_class]
    # ctx["dev"]     small dev set to score on
    # ctx["dataset"], ctx["rng"]
    # return a float; HIGHER = "more likely the best instruction".
    ...
```

## Datasets & Metric
- **agnews** - all 7,600 rows of the official AG News test split.
- **sst2** - all 872 labeled rows of the official SST-2 validation split.
Metric = zero-shot classification **accuracy** on the held-out TEST set of the
candidate your estimator ranks first, higher is better. Because ranking uses only DEV
but the score is on the disjoint TEST set, an estimator that chases dev noise or picks
a distractor scores near the class prior; a discriminating estimator generalizes.
The proposal pool (128 rows) and selection set (200 rows) come only from the
official training split and are pairwise disjoint from each other and from the
evaluation split. Only the single instruction selected on train/dev data is run on
the evaluation split. Both dataset settings participate in the score. Inference-only,
one GPU, with the two settings executed serially.
