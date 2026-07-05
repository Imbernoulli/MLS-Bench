# Text Simplification: Sampling vs. Beam-Search Decoding Strategy

## Research Question
Given a FROZEN small pretrained sentence-simplification model
(`mrm8488/t5-base-finetuned-turk-text-simplification`), which top-level **decoding
strategy** maximizes simplification quality (corpus **SARI**) across three distinct
test sets: plain multinomial **sampling**, nucleus (top-p) **sampling**, or
deterministic **beam search**? All three share the same fixed length window
(`max_length=128`) and the same frozen model; only the search strategy varies.

## Background
Sentence simplification rewrites a complex sentence into a simpler,
meaning-preserving one, scored by **SARI** (Xu et al. 2016):
`SARI = (F1_add + F1_keep + P_del)/3` over n=1..4. Beam search approximately
maximizes sequence probability by keeping the top-k partial hypotheses at each step;
sampling-based decoding (with or without nucleus truncation) never performs this
search, so it systematically under-performs beam search on a precision-sensitive
reference metric like SARI. This task compares the three strategies directly, on the
same model / same length window, isolating the STRATEGY choice from the more
fine-grained temperature / top-p / beam-width levers that sibling simp-* tasks vary
within each strategy family.

## Implementation Contract
Modify `text-simplification/solution/strategy.py`:

```python
def build_strategy() -> str:
    return "beam"
```

Must be one of `"sample"` (do_sample=True, num_beams=1, temperature=1.0),
`"topp"` (do_sample=True, num_beams=1, top_p=0.9), or
`"beam"` (num_beams=5, no_repeat_ngram_size=3). The model, the `simplify: ` prefix,
`max_length=128`, the three corpora, the references, the tokenizer, and the SARI
evaluator are all frozen in the harness.

## Fixed Pipeline & Evaluation
- Model: `mrm8488/t5-base-finetuned-turk-text-simplification` (T5-base, ~220M),
  FROZEN, eval mode.
- THREE FIXED test settings (from the ungated parquet `GEM/wiki_auto_asset_turk`,
  staged offline as JSONL `{source, references}`): `asset` (10 refs, multi-op),
  `turk` (7-8 refs, lexical), `wiki` (1 ref, real Wikipedia, longer sources).
- Metric (higher is better): `sari_{setting}` = corpus **SARI** (0-100); the task
  score is the geometric mean over the three settings. `bleu_{setting}` secondary.
- Deterministic given a fixed seed; runs on one small GPU in a few minutes.
- Scoring is a per-setting logistic anchored between the weak plain-sampling
  strategy and the strong beam-search strategy.
