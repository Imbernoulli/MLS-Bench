# Beam Width

## Research Question
Evaluate beam width for frozen, domain-matched summarization models. The same policy is evaluated on every example in three complete official test splits.

## Implementation Contract
Edit `abstractive-summarization/solution/beamwidth.py` and implement:

```python
def build_beam_width() -> int:
    ...
```

Return an integer within the runtime's documented compute bound. Missing keys, extra keys, invalid values, non-finite results, or
generation failures abort verification. The harness never repairs the editable
configuration or replaces it with another decode policy.

## Evaluation

All three required settings participate in the score:

- XSum official test: 11,334 documents.
- CNN/DailyMail 3.0.0 official test: 11,490 documents.
- SAMSum official test: 819 dialogues.

The verifier runs the pinned domain-matched summarizers in FP16 on one GPU,
serially over all 23,643 examples with generation batch size 16, and reports
mean per-example ROUGE-L F1 from Google's `rouge_score` implementation. For
every model-based policy, the tokenizer truncates each source to at most 512
subword tokens before padding to the longest source in that batch. This is a
fixed uniform comparison protocol, not a claim that 512 is the model's maximum
context and not a document subsample; every row in each official test split is
still evaluated. Non-model source policies, when selected in the source-policy
task, operate on every full source record according to the declared policy.

A non-zero score requires the task-specific surface proof, every dataset digest,
exact checkpoint revision, weight digest and parameter count, row and generation
count, finite bounded metric, ordered setting completion, and terminal final
proof to validate. Any command or verification failure receives zero.
