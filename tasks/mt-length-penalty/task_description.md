# Machine Translation: Length Normalization

## Research Question
How should beam hypotheses be length-normalized for a frozen translation model?
The penalty changes the balance between sequence probability and hypothesis length,
and therefore interacts with corpus brevity and precision.

## Implementation Contract
Modify `machine-translation/solution/length.py` so
`build_length_config() -> dict` returns exactly `length_penalty`, `min_length`,
and `max_new_tokens`. The penalty must be finite in `[0, 5]`; lengths must be
integers with `0 <= min_length <= max_new_tokens <= 160`. The beam width, model,
corpus, references, and evaluator are fixed.

## Fixed Evaluation
- Three pinned frozen OPUS-MT MarianMT checkpoints (~75M parameters each).
- The complete official 2000-pair OPUS-100 test split is used for each of three
  source languages, without filtering, sampling, shuffling, or head slicing.
- Corpus sacreBLEU is primary and chrF is secondary.
- Every direction executes serially on one GPU and contributes to the task score.
