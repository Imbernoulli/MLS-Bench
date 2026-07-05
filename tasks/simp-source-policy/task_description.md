# Text Simplification: Source Policy (monotonicity / anti-gaming)

## Research Question
Given a FROZEN small pretrained sentence-simplification model
(`mrm8488/t5-base-finetuned-turk-text-simplification`, a T5-base seq2seq simplifier
that takes a `simplify: <sentence>` input and emits a simpler paraphrase), which
**source policy** maximizes simplification quality (corpus **SARI**) across three
distinct test sets? The choices span meaning-destroying degenerate outputs (empty,
first-token, naive truncation) versus real model decodes (greedy vs a tuned beam).
This task verifies SARI is monotone and un-gameable: a meaning-destroying output
scores a genuinely LOW SARI on every setting, and only a real T5 simplifier reaches
the SOTA-scale top.

## Background
Sentence simplification rewrites a complex sentence into a simpler,
meaning-preserving one — via lexical paraphrasing, deletion of non-essential
content, and sentence splitting. It is distinct from summarization (compress),
machine translation (change language) and grammatical error correction (minimal
edits). Its canonical metric is **SARI** (Xu et al., TACL 2016):

    SARI = (F1_add + F1_keep + P_del) / 3   averaged over n = 1..4

SARI compares the SOURCE, the SYSTEM output, AND multiple human REFERENCES,
rewarding correct n-gram ADD / KEEP / DELETE edits. A meaning-destroying output
(empty / first-token) earns essentially no correct edits and scores a genuinely low
SARI (~11-20 across settings); a real simplifier that paraphrases and deletes the
right content reaches the SOTA scale (~43-45).

## Implementation Contract
Modify `text-simplification/solution/policy.py`:

```python
def build_policy() -> str:
    return "beam"   # empty | first_token | truncate | greedy | beam
```

Only the returned policy string varies. The model, the `simplify: ` prefix, the
three corpora, the references, the tokenizer, and the SARI evaluator are all frozen
in the harness. The `beam` policy uses a FIXED tuned config (beam 5, no-repeat-3gram)
you cannot change.

## Fixed Pipeline & Evaluation
- Model: `mrm8488/t5-base-finetuned-turk-text-simplification` (T5-base, ~220M
  params), FROZEN, eval mode; the `simplify: ` prefix is applied by the harness.
- THREE FIXED test settings (deterministic head-slice of each, from the ungated
  parquet dataset `GEM/wiki_auto_asset_turk`, staged offline as JSONL
  `{source, references}`):
  - `asset`: ASSET test — 10 multi-operation human refs/sentence (hardest).
  - `turk` : TurkCorpus test — 7-8 lexical refs/sentence.
  - `wiki` : WikiAuto manual test — 1 real-Wikipedia ref/sentence (longer sources).
- Metric (higher is better): `sari_{setting}` = corpus **SARI** (0-100). The task
  score is the geometric mean over the three settings, so a policy must simplify
  well across ALL THREE. `bleu_{setting}` (adequacy BLEU) is a secondary metric.
- Deterministic (greedy and beam search are deterministic under the frozen model);
  runs on one small GPU in a few minutes.
- Scoring is a per-setting logistic anchored between the meaning-destroying floor
  and the strong tuned-beam T5 decode. (Note: the pure copy-input identity baseline
  is intentionally NOT a selectable policy — on the conservative ASSET/Turk
  references it scores anomalously high SARI, a documented SARI-KEEP artifact, so a
  real DELETE/ADD-driven simplification is required to score well; the length /
  DELETE-balance lever is exercised by the sibling simp-length-control task.)
