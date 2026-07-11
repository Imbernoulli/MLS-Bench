# Machine Translation: Source Truncation

## Research Question
How many source subword tokens should be retained before encoding? A smaller window
reduces input work but can remove sentence content, while a larger window retains
more context up to the fixed model-side cap.

## Implementation Contract
Modify `machine-translation/solution/tok.py` so
`build_source_max_tokens() -> int` returns an integer from 1 through 128.
The tokenizer, generation policy, model, corpus, references, and evaluator are fixed.
Invalid values are rejected rather than clamped.

## Fixed Evaluation
- Three pinned frozen OPUS-MT MarianMT checkpoints (~75M parameters each).
- The complete official 2000-pair OPUS-100 test split is used for each of three
  source languages, without filtering, sampling, shuffling, or head slicing.
- Corpus sacreBLEU is primary and chrF is secondary.
- Every direction executes serially on one GPU and contributes to the task score.
