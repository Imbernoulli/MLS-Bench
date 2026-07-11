# Reasoning Budget for Numeric Decoding

## Research Question
Study the free-generation and answer token budgets for a frozen instruction model. The task measures the fraction of
examples whose committed answer is both structurally valid and correct. Evaluation
inputs and targets are fixed; targets are available only during verification.

## Implementation Contract
Edit `constrained-decoding-lab/solution/decoder_budget.py` and implement:

```python
def build_decoder(question, tok):
    ...
```

Return `common.DecodeSpec` with an explicit numeric-answer constraint and explicit positive integer token budgets. Missing fields, invalid types, non-finite model outputs, or runtime
exceptions fail verification; the harness does not substitute another decoder.

## Evaluation
The frozen model runs deterministically. Final assessment measures whether each
committed answer is both structurally valid and correct. Structural validity alone
is diagnostic and does not earn credit without a correct verifier-side target.
