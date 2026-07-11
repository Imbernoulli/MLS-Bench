# Reasoning Region for Forced-Choice Decoding

## Research Question
Study whether and how a free-form reasoning region precedes a constrained choice for a frozen instruction model. The task measures the fraction of
examples whose committed answer is both structurally valid and correct. Evaluation
inputs and targets are fixed; targets are available only during verification.

## Implementation Contract
Edit `constrained-decoding-lab/solution/decoder_choice_reasoning.py` and implement:

```python
def build_decoder(text, labels, tok):
    ...
```

Return `common.DecodeSpec` with a prompt and a choice constraint. Optional preamble/trigger fields and token budgets must satisfy the `DecodeSpec` contract. Missing fields, invalid types, non-finite model outputs, or runtime
exceptions fail verification; the harness does not substitute another decoder.

## Evaluation
The frozen model runs deterministically. Final assessment measures whether each
committed answer is both structurally valid and correct. Structural validity alone
is diagnostic and does not earn credit without a correct verifier-side target.
