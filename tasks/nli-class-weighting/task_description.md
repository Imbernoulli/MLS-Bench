# NLI Class Weighting

## Research Question
Study the class-weighting policy in a three-way natural-language-inference classifier. Training and evaluation use fixed real-corpus data. Evaluation rows and labels are staged only for verification.

## Implementation Contract
Edit `natural-language-inference/solution/class_weighting.py` and implement:

```python
def build_weighting() -> dict:
    ...
```

Return exactly a three-value `weights` vector in entailment, neutral,
contradiction order. Values must be finite numbers in `[0.25, 2.0]` with
arithmetic mean one, so the experiment changes relative class cost without
changing the overall loss scale. Missing or extra fields, invalid types,
non-finite logits/losses/gradients, or incomplete evaluation outputs fail the run.
No implementation or selector default is substituted.

## Evaluation
The fixed training loop reports three-way accuracy for each configured evaluation. Every configured evaluation must finish with complete, finite predictions for the task to receive a
non-zero score.
