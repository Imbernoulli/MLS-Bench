# Text Simplification: Minimum-Length Floor (Isolated)

## Research Question
Given a FROZEN small pretrained sentence-simplification model
(`mrm8488/t5-base-finetuned-turk-text-simplification`, a T5-base seq2seq simplifier
taking `simplify: <sentence>`) decoded under a FIXED beam width, length-penalty and
max_length (`num_beams=5`, `no_repeat_ngram_size=3`, `length_penalty=1.0`,
`max_length=96` all FIXED), how should the decoder-side **min_length FLOOR** (the
minimum number of generated tokens before EOS is allowed) be set to maximize
simplification quality (corpus **SARI**) across three distinct test sets? This
isolates the min-length floor from the length window's upper bound (`max_length`,
governed by `simp-length-control`) and from `length_penalty` (governed by
`simp-decoding-temperature`'s neighbor tasks).

## Background
Sentence simplification often SHORTENS a sentence (drops subordinate clauses,
omits/splits complex material). **SARI** (Xu et al. 2016),
`SARI = (F1_add + F1_keep + P_del)/3` over n=1..4, rewards correct DELETE edits. A
large `min_length` FLOOR forces the decoder to keep generating past the point where
it would naturally stop at EOS, padding the output with low-value continuation
tokens that rarely correspond to a correct ADD/KEEP edit and push the sequence away
from a genuine compressive simplification. A small/zero floor lets the model stop
wherever beam search naturally finds the best EOS, matching `simp-length-control`'s
tuned config (`min_length=0`).

## Implementation Contract
Modify `text-simplification/solution/minlen.py`:

```python
def build_min_length() -> int:
    return 0
```

Hard-capped to `[0, 96]` (clamped to `max_length`) by the shared sanitizer. The
model, the `simplify: ` prefix, `num_beams=5`, `no_repeat_ngram_size=3`,
`length_penalty=1.0`, `max_length=96`, the three corpora, the references, the
tokenizer, and the SARI evaluator are all frozen in the harness.

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
- Scoring is a per-setting logistic anchored between the weak large floor (60) and
  the strong zero floor (0).
