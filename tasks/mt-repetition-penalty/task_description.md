# Machine Translation: Token Repetition Penalty

## Research Question
What token-level repetition penalty gives the best translation quality under fixed
beam decoding? The penalty changes logits for tokens already generated and can alter
both unwanted loops and legitimate repeated words.

## Implementation Contract
Modify `machine-translation/solution/reppen.py` so
`build_reppen_config() -> dict` returns exactly
`{"repetition_penalty": value}`, where `value` is finite and lies in `[0.1, 5]`.
The beam policy, model, corpus, references, and evaluator are fixed.

## Fixed Evaluation
- Three pinned frozen OPUS-MT MarianMT checkpoints (~75M parameters each).
- The complete official 2000-pair OPUS-100 test split is used for each of three
  source languages, without filtering, sampling, shuffling, or head slicing.
- Corpus sacreBLEU is primary and chrF is secondary.
- Every direction executes serially on one GPU and contributes to the task score.
