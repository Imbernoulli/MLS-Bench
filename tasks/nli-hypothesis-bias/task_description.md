# NLI Premise Usage

## Research Question
Study whether the model may use the premise in a three-way natural-language-inference classifier. Training and evaluation use fixed real-corpus data. Evaluation rows and labels are staged only for verification.

## Implementation Contract
Edit `natural-language-inference/solution/hypothesis_bias.py` and implement:

```python
def build_bias() -> dict:
    ...
```

Return exactly `use_premise` as a boolean, or the explicitly supported diagnostic mode. Missing or extra fields, invalid selector values, invalid types,
non-finite logits/losses/gradients, or incomplete evaluation outputs fail the run.
No implementation or selector default is substituted.

## Evaluation
The fixed training loop reports three-way accuracy for each configured evaluation. Every configured evaluation must finish with complete, finite predictions for the task to receive a
non-zero score.
