# Machine Translation: Beam Search and Repetition Control

## Research Question
How should beam width and repeated n-gram blocking be configured for a frozen
sequence-to-sequence translation model? Beam width changes search error and length
bias, while repeated n-gram blocking can prevent loops but can also reject legitimate
repetition.

## Implementation Contract
Modify `machine-translation/solution/beam.py` so `build_beam_config() -> dict`
returns exactly:

```python
{"num_beams": <integer 1..12>, "no_repeat_ngram_size": <integer 0..10>}
```

The length policy, source handling, models, references, and evaluator are fixed.
Values outside the declared ranges are rejected rather than clamped.

## Fixed Evaluation
- Three pinned frozen OPUS-MT MarianMT checkpoints (~75M parameters each).
- The complete official 2000-pair OPUS-100 test split is used for each of three
  source languages, without filtering, sampling, shuffling, or head slicing.
- Corpus sacreBLEU is primary and chrF is secondary.
- Every direction executes serially on one GPU and contributes to the task score.
