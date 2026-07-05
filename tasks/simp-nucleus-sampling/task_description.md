# Text Simplification: Nucleus (Top-p) Sampling

## Research Question
Given a FROZEN small pretrained sentence-simplification model
(`mrm8488/t5-base-finetuned-turk-text-simplification`), decoded via SAMPLING
(`do_sample=True`, `num_beams=1` FIXED, `temperature=1.0` FIXED — no beam search, no
temperature reshaping), how should the **nucleus (top-p)** mass be set to maximize
simplification quality (corpus **SARI**) across three distinct test sets? Nucleus
sampling (Holtzman et al. 2019) keeps the smallest token set whose cumulative
probability >= `top_p`, renormalizes, and samples only from that set.

## Background
Sentence simplification rewrites a complex sentence into a simpler,
meaning-preserving one, scored by **SARI** (Xu et al. 2016):
`SARI = (F1_add + F1_keep + P_del)/3` over n=1..4. A WIDE nucleus (`top_p` close to
1.0) samples from (nearly) the full vocabulary, including many low-probability /
off-distribution tokens that rarely land on a correct simplified phrasing, hurting
SARI. A TIGHT nucleus restricts sampling to only the model's most probable tokens,
staying closer to a faithful (near-greedy) simplification and improving SARI. This
isolates the top-p lever from temperature and beam search (both FIXED).

## Implementation Contract
Modify `text-simplification/solution/nucleus.py`:

```python
def build_top_p() -> float:
    return 0.6
```

`top_p` is hard-capped to `[0.01, 1.0]` by the shared sanitizer. The model, the
`simplify: ` prefix, `do_sample=True`, `num_beams=1`, `temperature=1.0`,
`no_repeat_ngram_size=3`, `max_length=128`, the three corpora, the references, the
tokenizer, and the SARI evaluator are all frozen in the harness.

## Fixed Pipeline & Evaluation
- Model: `mrm8488/t5-base-finetuned-turk-text-simplification` (T5-base, ~220M),
  FROZEN, eval mode.
- THREE FIXED test settings (from the ungated parquet `GEM/wiki_auto_asset_turk`,
  staged offline as JSONL `{source, references}`): `asset` (10 refs, multi-op),
  `turk` (7-8 refs, lexical), `wiki` (1 ref, real Wikipedia, longer sources).
- Metric (higher is better): `sari_{setting}` = corpus **SARI** (0-100); the task
  score is the geometric mean over the three settings. `bleu_{setting}` secondary.
- Deterministic given a fixed seed; runs on one small GPU in a few minutes.
- Scoring is a per-setting logistic anchored between the weak wide-nucleus sample
  and the strong tight-nucleus sample.
