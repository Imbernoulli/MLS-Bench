# Machine Translation: Output-Length Budget

## Research Question
What output token budget gives the best corpus translation quality while allowing
the decoder to finish naturally? A short budget can truncate valid translations,
while a larger budget increases worst-case decoding work and can permit unwanted
continuations.

## Implementation Contract
Modify `machine-translation/solution/maxlen.py` so
`build_max_new_tokens() -> int` returns an integer from 1 through 160. The model,
beam policy, source handling, references, and evaluator are fixed.

## Fixed Evaluation
- Three pinned frozen OPUS-MT MarianMT checkpoints (~75M parameters each).
- The complete official 2000-pair OPUS-100 test split is used for each of three
  source languages, without filtering, sampling, shuffling, or head slicing.
- Corpus sacreBLEU is primary and chrF is secondary.
- Every direction executes serially on one GPU and contributes to the task score.
