# Machine Translation: Repeated N-gram Blocking

## Research Question
What repeated n-gram blocking order gives the best translation quality under a
fixed beam and length policy? Smaller orders impose a stronger lexical constraint;
larger orders intervene only on longer repeated spans.

## Implementation Contract
Modify `machine-translation/solution/norep.py` so
`build_norep_config() -> dict` returns exactly
`{"no_repeat_ngram_size": n}`, where `n` is an integer from 0 through 10.
The beam policy, model, corpus, references, and evaluator are fixed.

## Fixed Evaluation
- Three pinned frozen OPUS-MT MarianMT checkpoints (~75M parameters each).
- The complete official 2000-pair OPUS-100 test split is used for each of three
  source languages, without filtering, sampling, shuffling, or head slicing.
- Corpus sacreBLEU is primary and chrF is secondary.
- Every direction executes serially on one GPU and contributes to the task score.
