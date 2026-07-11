# Machine Translation: Search Policy

## Research Question
How do stochastic sampling and deterministic search compare for corpus translation
quality under a frozen model? Choose among ancestral sampling, nucleus sampling,
greedy search, and beam search while all other evaluation inputs remain fixed.

## Implementation Contract
Modify `machine-translation/solution/sampling.py` so
`build_mode() -> str` returns one of:

`sample_t1` | `topp` | `greedy` | `beam`

Each mode maps to a fixed generation configuration. The model, corpus, references,
tokenizer, and evaluator are fixed; stochastic modes use the declared seed.

## Fixed Evaluation
- Three pinned frozen OPUS-MT MarianMT checkpoints (~75M parameters each).
- The complete official 2000-pair OPUS-100 test split is used for each of three
  source languages, without filtering, sampling, shuffling, or head slicing.
- Corpus sacreBLEU is primary and chrF is secondary.
- Every direction executes serially on one GPU and contributes to the task score.
