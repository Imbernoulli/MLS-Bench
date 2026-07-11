# NLI Encoder Update Policy

## Research Question
Study whether and how the encoder is updated in a three-way
natural-language-inference classifier. The fixed training and assessment pipeline
uses real-corpus data; assessment rows and labels remain outside the editable
workspace.

## Implementation Contract
Edit `natural-language-inference/solution/finetune.py` and implement:

```python
def build_finetune() -> dict:
    ...
```

Return exactly an `encoder` selector supported by the interface. Missing or extra fields, invalid selector values, invalid types,
non-finite logits/losses/gradients, or incomplete evaluation outputs fail the run.
No implementation or selector default is substituted.

## Evaluation
The fixed runner uses every labeled SNLI training row for three epochs, then
reports three-way accuracy on the complete labeled SNLI test, MultiNLI matched
development, and MultiNLI mismatched development splits. Every training epoch,
optimizer step inventory, and configured evaluation must finish with complete,
finite results for the task to receive a non-zero score.
