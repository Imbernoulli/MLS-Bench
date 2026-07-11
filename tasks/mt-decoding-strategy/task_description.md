# Machine Translation: Decode Policy

## Research Question
Which decode policy produces the best corpus translation quality under a frozen
translation model? Compare model-backed search policies with control policies that
copy or truncate the source. The controls test whether the metric tracks actual
translation rather than format-only output.

## Implementation Contract
Modify `machine-translation/solution/strategy.py` so
`build_strategy() -> str` returns one of:

`beam` | `greedy` | `copy_source` | `first_token` | `empty`

The model-backed generation configurations, corpus, references, tokenizer, and
evaluator are fixed.

## Fixed Evaluation
- Three pinned frozen OPUS-MT MarianMT checkpoints (~75M parameters each).
- The complete official 2000-pair OPUS-100 test split is used for each of three
  source languages, without filtering, sampling, shuffling, or head slicing.
- Corpus sacreBLEU is primary and chrF is secondary.
- Every direction executes serially on one GPU and contributes to the task score.
