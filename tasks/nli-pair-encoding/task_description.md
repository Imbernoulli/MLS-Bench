# NLI Pair Encoding

## Research Question
Compare a joint cross-encoder with a fixed mean-pooled InferSent-style bi-encoder architecture for three-way natural-language inference. Training and evaluation use fixed real-corpus data. Evaluation rows and labels are staged only for verification.

## Implementation Contract
Edit `natural-language-inference/solution/pair_encoding.py` and implement:

```python
def build_encoding() -> dict:
    ...
```

Return exactly an `encoding` selector supported by the interface. Missing or extra fields, invalid selector values, invalid types,
non-finite logits/losses/gradients, or incomplete evaluation outputs fail the run.
No implementation or selector default is substituted.

## Evaluation
The fixed training loop reports three-way accuracy for each configured evaluation. Every configured evaluation must finish with complete, finite predictions for the task to receive a
non-zero score.
