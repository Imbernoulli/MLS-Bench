# Automatic Prompt Optimization: Instruction Search

## Research Question
A **frozen** small instruction LM (Qwen2.5-0.5B-Instruct) does **zero-shot** text
classification — it is shown ONLY an instruction and the input, and predicts the
label word it assigns the highest *contextually-calibrated* per-label likelihood (no
demonstrations in the executed prompt). The instruction text alone strongly
determines accuracy. This is
**Automatic Prompt Engineering** (APE): treat the instruction as a program to be
searched. Design `optimize(ctx)` to PROPOSE candidate instructions and SELECT one
that maximizes accuracy — but you are scored on a **held-out TEST set**, so a
candidate overfit to the small dev set will not win.

This is a DISTINCT axis from in-context-learning (the icl-* tasks): there the
demonstrations are selected/ordered/calibrated and the instruction is FIXED; here the
demonstrations are FIXED/ABSENT and you search the **INSTRUCTION TEXT**.

## Background
Zhou et al., "Large Language Models Are Human-Level Prompt Engineers" (ICLR 2023,
[arXiv:2211.01910](https://arxiv.org/abs/2211.01910)) cast prompt engineering as
instruction induction + search: an LM PROPOSES candidate instructions from a few
input/output examples (reverse-mode induction, Honovich et al. 2022,
[arXiv:2205.10782](https://arxiv.org/abs/2205.10782)), each candidate is SCORED by its
**execution accuracy** on a dev set, and the best is kept. APE-generated instructions
match or beat human prompts on 24/24 instruction-induction tasks. The key discipline
is generalization: select on dev, but the instruction that wins should transfer to
unseen test inputs.

Reference baselines (provided as read-only edit ops):
- **empty** — the empty/zero instruction (no task guidance; near the class prior).
- **ape** — LM-induce candidates from the pool + a hand-written anchor, select the
  highest dev-accuracy candidate (Zhou et al. 2022).

## What is FIXED (you cannot change)
The base LM, the contextually-calibrated zero-shot forced-choice executor, the
proposal POOL, the DEV set, the HELD-OUT TEST set, and the label set. You control
**only** the instruction string returned by `optimize` (and how you propose/select
it).

## Model Interface
Implement `optimize` in `prompt-optimization-lab/solution/search.py`:
```python
def optimize(ctx) -> str:
    # ctx["executor"].dev_accuracy(instr, dev_rows) -> float
    # ctx["executor"].predict(instr, rows) -> label ids
    # ctx["pool"]  labeled examples to INDUCE from
    # ctx["dev"]   small dev set to SELECT on
    # ctx["induce_instructions"](n) -> list[str]  LM-induced candidates
    # ctx["rng"], ctx["dataset"]
    # return ONE instruction string (may be "").
    ...
```

## Datasets & Metric
- **agnews** — AG News, 4-class topic classification (majority floor 0.25).
- **sst2** — SST-2, 2-class sentiment (majority floor 0.50; official evaluation split).
Metric = zero-shot classification **accuracy** of the chosen instruction on a fixed
held-out TEST set (300 examples), higher is better. An empty/degenerate instruction
sits near the class prior; a well-searched, task-specific instruction scores clearly
higher and generalizes from dev to test. Inference-only, single GPU, a few minutes.

## Evaluation Protocol
- **agnews** - all 7,600 rows of the official AG News test split.
- **sst2** - all 872 labeled rows of the official SST-2 validation split.

The proposal pool (128 rows) and candidate-selection set (200 rows) are drawn only
from the official training split. Their texts are disjoint from each other and from
the evaluation split. Candidate generation and ranking use only train/dev data; the
verifier runs exactly one selected instruction on the complete evaluation split.
Both dataset settings participate in the score. The settings run serially on one GPU.
