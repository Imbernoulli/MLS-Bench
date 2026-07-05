# Text Simplification: Sampling Temperature

## Research Question
Given a FROZEN small pretrained sentence-simplification model
(`mrm8488/t5-base-finetuned-turk-text-simplification`, a T5-base seq2seq simplifier
taking `simplify: <sentence>`), decoded via SAMPLING (`do_sample=True`, `num_beams=1`
FIXED, `no_repeat_ngram_size=3` FIXED — no beam search), how should the softmax
**temperature** be set to maximize simplification quality (corpus **SARI**) across
three distinct test sets? Temperature reshapes the softmax before sampling: a HOT
temperature (>1) flattens the distribution towards uniform, drawing more
off-distribution tokens; a COLD temperature (<1) sharpens it towards the model's
mode, drawing the model's preferred tokens more often.

## Background
Sentence simplification rewrites a complex sentence into a simpler,
meaning-preserving one, scored by **SARI** (Xu et al. 2016):
`SARI = (F1_add + F1_keep + P_del)/3` over n=1..4, which compares the SOURCE, the
SYSTEM output and multiple human REFERENCES and rewards correct ADD/KEEP/DELETE
n-gram edits. Sampling with no search (`num_beams=1`) never performs a beam-search
optimization, so among sampling configurations, temperature is the direct lever on
how closely the sampled tokens track the model's own (fine-tuned, meaning-preserving)
distribution: hotter sampling draws more random, less faithful tokens (lower SARI);
colder sampling draws the model's preferred tokens more often (higher SARI, closer
to greedy).

## Implementation Contract
Modify `text-simplification/solution/temperature.py`:

```python
def build_temperature() -> float:
    return 1.0
```

`temperature` is hard-capped to `[0.05, 2.5]` by the shared sanitizer. The model, the
`simplify: ` prefix, `do_sample=True`, `num_beams=1` (no beam search),
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
- Scoring is a per-setting logistic anchored between the weak hot-temperature
  sample and the strong cold-temperature sample.
