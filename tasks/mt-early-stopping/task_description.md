# Machine Translation: Beam Early-Stopping Policy

## Research Question
When should beam search terminate under a fixed beam and length-normalization policy?
Compare immediate completion after enough finished hypotheses, heuristic completion,
and canonical completion only when no better hypothesis can remain.

## Implementation Contract
Modify `machine-translation/solution/earlystop.py` so
`build_early_stopping()` returns exactly `True`, `False`, or `"never"`.
The beam width, length policy, model, corpus, references, and evaluator are fixed.

## Fixed Evaluation
- Three pinned frozen OPUS-MT MarianMT checkpoints (~75M parameters each).
- The complete official 2000-pair OPUS-100 test split is used for each of three
  source languages, without filtering, sampling, shuffling, or head slicing.
- Corpus sacreBLEU is primary and chrF is secondary.
- Every direction executes serially on one GPU and contributes to the task score.
