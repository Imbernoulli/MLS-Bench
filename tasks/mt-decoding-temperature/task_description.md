# Machine Translation: Sampling Temperature

## Research Question
Under a fixed nucleus-sampling policy, what softmax temperature gives the best
corpus translation quality? Temperature changes the concentration and diversity of
the token distribution, so the useful value depends on model calibration and search
stochasticity.

## Implementation Contract
Modify `machine-translation/solution/temperature.py` so
`build_temperature() -> float` returns a finite number in `[0.05, 5.0]`.
The nucleus threshold, model, corpus, references, tokenizer, and evaluator are fixed.
Invalid values are rejected rather than clamped.

## Fixed Evaluation
- Three pinned frozen OPUS-MT MarianMT checkpoints (~75M parameters each).
- The complete official 2000-pair OPUS-100 test split is used for each of three
  source languages, without filtering, sampling, shuffling, or head slicing.
- Corpus sacreBLEU is primary and chrF is secondary.
- Every direction executes serially on one GPU and contributes to the task score.
