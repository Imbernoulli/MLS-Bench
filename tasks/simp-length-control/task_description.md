# Text Simplification: Length / Compression Control

## Research Question
Given a FROZEN small pretrained sentence-simplification model
(`mrm8488/t5-base-finetuned-turk-text-simplification`, a T5-base seq2seq simplifier
taking `simplify: <sentence>`) decoded with a FIXED strong beam (num_beams=5,
no-repeat-3gram), how should the **length / compression window** (`min_length`,
`max_length`, `length_penalty`) be configured to maximize simplification quality
(corpus **SARI**) across three distinct test sets?

## Background
Sentence simplification often SHORTENS a sentence (drops subordinate clauses,
omits/splits complex material). **SARI** (Xu et al. 2016),
`SARI = (F1_add + F1_keep + P_del)/3` over n=1..4, rewards correct DELETE edits, so
length is a direct lever on the DELETE/ADD balance. A runaway-long decode (large
`length_penalty`, large `max_length`) keeps everything — behaving like
copy-the-input (few DELETE credits -> lower SARI). An over-short decode drops too
much (recall of the right content collapses). A sensibly compressive window near the
reference length maximizes SARI.

## Implementation Contract
Modify `text-simplification/solution/length.py`:

```python
def build_length_config() -> dict:
    return {"min_length": 0, "max_length": 96, "length_penalty": 1.0}
```

`max_length` is hard-capped at 200. `length_penalty > 1.0` favours LONGER outputs,
`< 1.0` favours SHORTER (more compression). The model, the `simplify: ` prefix, the
FIXED beam (num_beams=5, no_repeat_ngram_size=3), the three corpora, the references,
the tokenizer, and the SARI evaluator are all frozen in the harness.

## Fixed Pipeline & Evaluation
- Model: `mrm8488/t5-base-finetuned-turk-text-simplification` (T5-base, ~220M),
  FROZEN, eval mode.
- THREE FIXED test settings (from the ungated parquet `GEM/wiki_auto_asset_turk`,
  staged offline as JSONL `{source, references}`): `asset` (10 refs, multi-op),
  `turk` (7-8 refs, lexical), `wiki` (1 ref, real Wikipedia, longer sources).
- Metric (higher is better): `sari_{setting}` = corpus **SARI** (0-100); the task
  score is the geometric mean over the three settings. `bleu_{setting}` secondary.
- Deterministic; runs on one small GPU in a few minutes.
- Scoring is a per-setting logistic anchored between the weak runaway-long window
  (under-compressed, near copy-input) and the tuned compressive window.
