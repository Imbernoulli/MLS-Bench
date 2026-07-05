# Text Simplification: Beam Width (Isolated)

## Research Question
Given a FROZEN small pretrained sentence-simplification model
(`mrm8488/t5-base-finetuned-turk-text-simplification`, a T5-base seq2seq simplifier
taking `simplify: <sentence>`) decoded under a FIXED repetition/length config
(`no_repeat_ngram_size=3`, `length_penalty=1.0`, `max_length=128`,
`early_stopping=True` all FIXED), how should the **beam WIDTH** (`num_beams`) be set
to maximize simplification quality (corpus **SARI**) across three distinct test
sets? This isolates the beam-width lever alone, distinct from `simp-decoding-beam`
(which varies `num_beams` and `no_repeat_ngram_size` jointly).

## Background
Sentence simplification rewrites a complex sentence into a simpler,
meaning-preserving one, scored by **SARI** (Xu et al. 2016):
`SARI = (F1_add + F1_keep + P_del)/3` over n=1..4. Beam search approximately
maximizes sequence probability by keeping the top-`k` partial hypotheses at each
decoding step; a NARROW beam (2) under-searches and settles for a lower-probability
(and lower-SARI) completion, while a WIDER beam (4, 8) explores more candidate
continuations and more reliably finds the sequence with the best ADD/KEEP/DELETE
edits, up to diminishing returns.

## Implementation Contract
Modify `text-simplification/solution/beamwidth.py`:

```python
def build_num_beams() -> int:
    return 8
```

Hard-capped to `[1, 12]` by the shared sanitizer. The model, the `simplify: `
prefix, `no_repeat_ngram_size=3`, `length_penalty=1.0`, `max_length=128`,
`early_stopping=True`, the three corpora, the references, the tokenizer, and the
SARI evaluator are all frozen in the harness.

## Fixed Pipeline & Evaluation
- Model: `mrm8488/t5-base-finetuned-turk-text-simplification` (T5-base, ~220M),
  FROZEN, eval mode.
- THREE FIXED test settings (from the ungated parquet `GEM/wiki_auto_asset_turk`,
  staged offline as JSONL `{source, references}`): `asset` (10 refs, multi-op),
  `turk` (7-8 refs, lexical), `wiki` (1 ref, real Wikipedia, longer sources).
- Metric (higher is better): `sari_{setting}` = corpus **SARI** (0-100); the task
  score is the geometric mean over the three settings. `bleu_{setting}` secondary.
- Deterministic (beam search does not depend on the RNG seed); runs on one small
  GPU in a few minutes.
- Scoring is a per-setting logistic anchored between the weak narrow beam (2) and
  the strong wide beam (8).
