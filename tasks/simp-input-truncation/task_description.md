# Text Simplification: Encoder-Side Input-Truncation Budget

## Research Question
Given a FROZEN small pretrained sentence-simplification model
(`mrm8488/t5-base-finetuned-turk-text-simplification`) decoded under a FIXED strong
beam-search config (`num_beams=5`, `no_repeat_ngram_size=3`,
`length_penalty=1.0`, `max_length=128` all FIXED), how should the **encoder-side
input-truncation budget** (the tokenizer's `max_length` / `truncation=True` applied
to the SOURCE before encoding — NOT the decoder-side generation length) be set to
maximize simplification quality (corpus **SARI**) across three distinct test sets?

## Background
Sentence simplification rewrites a complex sentence into a simpler,
meaning-preserving one, scored by **SARI** (Xu et al. 2016):
`SARI = (F1_add + F1_keep + P_del)/3` over n=1..4. Text-simplification sources are
short sentences (mean ~15-25 words for asset/turk; WikiAuto sources run longer, up
to 80 words = 100+ subword tokens): an AGGRESSIVELY SHORT input budget silently
drops the tail of longer sources before the model ever sees it, and the model then
has no way to recover the deleted content's ADD/KEEP credit — SARI drops noticeably,
especially on the longer `wiki` setting. A moderate budget lets the model read
enough of each source to simplify it faithfully. This isolates the ENCODER-side
truncation lever from the (FIXED) decode config used by every other simp-* task —
a distinct failure mode from `length_penalty` / `max_length` (those govern the
OUTPUT; this governs how much of the INPUT the model ever sees).

## Implementation Contract
Modify `text-simplification/solution/truncation.py`:

```python
def build_max_input_tokens() -> int:
    return 48
```

Hard-capped to `[8, 160]` (the harness's `MAX_INPUT_TOKENS`). The model, the
`simplify: ` prefix, the FIXED beam/repetition/length-penalty decode config, the
three corpora, the references, the tokenizer, and the SARI evaluator are all frozen
in the harness.

## Fixed Pipeline & Evaluation
- Model: `mrm8488/t5-base-finetuned-turk-text-simplification` (T5-base, ~220M),
  FROZEN, eval mode.
- THREE FIXED test settings (from the ungated parquet `GEM/wiki_auto_asset_turk`,
  staged offline as JSONL `{source, references}`): `asset` (10 refs, multi-op),
  `turk` (7-8 refs, lexical), `wiki` (1 ref, real Wikipedia, longer sources).
- Metric (higher is better): `sari_{setting}` = corpus **SARI** (0-100); the task
  score is the geometric mean over the three settings. `bleu_{setting}` secondary.
- Deterministic; runs on one small GPU in a few minutes.
- Scoring is a per-setting logistic anchored between the weak aggressively-short
  input budget and the strong moderate input budget.
