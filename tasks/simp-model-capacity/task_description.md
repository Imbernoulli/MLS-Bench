# Text Simplification: Model Capacity / Checkpoint Choice

## Research Question
Holding a FIXED strong beam-search decode config (`num_beams=5`,
`no_repeat_ngram_size=3`, `length_penalty=1.0`, `max_length=128`) constant, which of
three FROZEN, staged-offline pretrained sentence-simplification checkpoints (all
from the `wiki_auto_asset_turk` fine-tune family) should be selected to maximize
simplification quality (corpus **SARI**) across three distinct test sets?

## Background
Sentence simplification rewrites a complex sentence into a simpler,
meaning-preserving one, scored by **SARI** (Xu et al. 2016):
`SARI = (F1_add + F1_keep + P_del)/3` over n=1..4. Holding the decode config fixed,
MORE MODEL CAPACITY (t5-base, ~220M params, vs t5-small, 60M params — 3.7x the
parameters) is the standard "bigger backbone helps" lever in seq2seq NLP. This task
isolates the checkpoint-choice lever: NONE of the three checkpoints are trained here
— this selects among EXISTING community checkpoints, all originally fine-tuned on
subsets of the same `wiki_auto_asset_turk` corpus family, so the comparison holds
the decode config, the source data, and the references fixed and varies only WHICH
frozen model does the rewriting.

## Implementation Contract
Modify `text-simplification/solution/capacity.py`:

```python
def build_model_choice() -> str:
    return "base_turk"
```

Must be one of the three staged checkpoints:
- `"small_wikiauto"` : t5-small-finetuned-text-simplification (t5-small, 60M params,
  broader wiki_auto_asset_turk fine-tune mix).
- `"small_turk"` : t5-small-finetuned-turk-text-simplification (t5-small, 60M params,
  mainly TurkCorpus-style lexical edits).
- `"base_turk"` : t5-base-finetuned-turk-text-simplification (t5-base, ~220M params —
  the model used by every other simp-* task).

The FIXED strong beam decode config, the three corpora, the references, the
tokenizer, and the SARI evaluator are all frozen in the harness.

## Fixed Pipeline & Evaluation
- Model: agent-chosen one of the three FROZEN checkpoints above, eval mode.
- THREE FIXED test settings (from the ungated parquet `GEM/wiki_auto_asset_turk`,
  staged offline as JSONL `{source, references}`): `asset` (10 refs, multi-op),
  `turk` (7-8 refs, lexical), `wiki` (1 ref, real Wikipedia, longer sources).
- Metric (higher is better): `sari_{setting}` = corpus **SARI** (0-100); the task
  score is the geometric mean over the three settings. `bleu_{setting}` secondary.
- Deterministic; runs on one small GPU in a few minutes.
- Scoring is a per-setting logistic anchored between the weakest
  (`small_wikiauto`) and strongest (`base_turk`) measured checkpoint.
