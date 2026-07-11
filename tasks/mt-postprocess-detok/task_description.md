# Machine Translation: Output Post-processing

## Research Question
Which deterministic output-normalization policy best matches the reference-text
convention used by a standard corpus translation metric? Post-processing can repair
spacing, preserve the decoded surface, or remove case and punctuation information.

## Implementation Contract
Modify `machine-translation/solution/postproc.py` so
`build_postproc() -> str` returns one of:

`normalize` | `lowercase` | `strip_punct`

The model output before post-processing, corpus, references, tokenizer, and evaluator
are fixed.

## Fixed Evaluation
- Three pinned frozen OPUS-MT MarianMT checkpoints (~75M parameters each).
- The complete official 2000-pair OPUS-100 test split is used for each of three
  source languages, without filtering, sampling, shuffling, or head slicing.
- Corpus sacreBLEU is primary and chrF is secondary.
- Every direction executes serially on one GPU and contributes to the task score.
