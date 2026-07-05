# Text Simplification: Beam Search & Repetition Decoding

## Research Question
Given a FROZEN small pretrained sentence-simplification model
(`mrm8488/t5-base-finetuned-turk-text-simplification`, a T5-base seq2seq simplifier
taking `simplify: <sentence>`), how should the **beam search** and **repetition
control** be configured to maximize simplification quality (corpus **SARI**) across
three distinct test sets? The classic choice is **greedy** decoding (`num_beams=1`)
versus **beam search** (`num_beams` 4-6), with a small `no_repeat_ngram_size` to
block degenerate repetition. The length window is FIXED here so only the
beam/repetition config varies.

## Background
Sentence simplification rewrites a complex sentence into a simpler,
meaning-preserving one (lexical paraphrase + deletion + splitting), scored by
**SARI** (Xu et al. 2016): `SARI = (F1_add + F1_keep + P_del)/3` over n=1..4, which
compares the SOURCE, the SYSTEM output and multiple human REFERENCES and rewards
correct ADD/KEEP/DELETE n-gram edits. Beam search approximately maximizes sequence
probability by keeping the top-`k` partial hypotheses; greedy under-searches and
misses the paraphrase/delete edits a moderate beam recovers. A small
`no_repeat_ngram_size` (2-3) avoids repetition loops without hurting real edits.

## Implementation Contract
Modify `text-simplification/solution/beam.py`:

```python
def build_beam_config() -> dict:
    return {"num_beams": 5, "no_repeat_ngram_size": 3, "repetition_penalty": 1.0}
```

`num_beams` is hard-capped at 12. The model, the `simplify: ` prefix, the FIXED
length window (`max_length=128`, `length_penalty=1.0`), the three corpora, the
references, the tokenizer, and the SARI evaluator are all frozen in the harness.

## Fixed Pipeline & Evaluation
- Model: `mrm8488/t5-base-finetuned-turk-text-simplification` (T5-base, ~220M),
  FROZEN, eval mode.
- THREE FIXED test settings (from the ungated parquet `GEM/wiki_auto_asset_turk`,
  staged offline as JSONL `{source, references}`): `asset` (10 refs, multi-op),
  `turk` (7-8 refs, lexical), `wiki` (1 ref, real Wikipedia, longer sources).
- Metric (higher is better): `sari_{setting}` = corpus **SARI** (0-100); the task
  score is the geometric mean over the three settings. `bleu_{setting}` secondary.
- Deterministic; runs on one small GPU in a few minutes.
- Scoring is a per-setting logistic anchored between the weak greedy decode and the
  strong tuned-beam decode.
