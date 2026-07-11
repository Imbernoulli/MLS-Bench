# Grouped Beam Configuration

## Research Question
Evaluate beam grouping and diversity penalty for frozen, domain-matched summarization models. The same policy is evaluated on every example in three complete official test splits.

## Implementation Contract
Edit `abstractive-summarization/solution/diverse.py` and implement:

```python
def build_diverse_config() -> dict:
    ...
```

Return exactly `num_beams`, `num_beam_groups`, and `diversity_penalty`; groups must divide the beam count. Missing keys, extra keys, invalid values, non-finite results, or
generation failures abort verification. The harness never repairs the editable
configuration or replaces it with another decode policy.

## Evaluation

All three required settings participate in the score:

- XSum official test: 11,334 documents.
- CNN/DailyMail 3.0.0 official test: 11,490 documents.
- SAMSum official test: 819 dialogues.

The verifier runs the pinned domain-matched summarizers in FP16 on one GPU, serially over all 23,643 examples, and reports corpus ROUGE-L F1. A non-zero score requires every dataset digest, model digest, row count, generation count, finite metric, setting completion, and final completion proof to validate. Any command or verification failure receives zero.