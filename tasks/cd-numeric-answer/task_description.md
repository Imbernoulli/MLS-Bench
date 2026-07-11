# Constrained Decoding for Numeric Answers

## Research Question
Study which part of generation is constrained and how the numeric answer region is defined for a frozen instruction model. The task measures the fraction of
examples whose committed answer is both structurally valid and correct. Evaluation
inputs and targets are fixed; targets are available only during verification.

## Implementation Contract
Edit `constrained-decoding-lab/solution/decoder_numeric.py` and implement:

```python
def build_decoder(question, tok):
    ...
```

Return `common.DecodeSpec` with a prompt and exactly one answer constraint. A preamble and trigger may be used together; token budgets must satisfy the runtime contract. Missing fields, invalid types, non-finite model outputs, or runtime
exceptions fail verification; the harness does not substitute another decoder.

## Evaluation
The frozen model runs deterministically and reports answer accuracy. Structural
validity is diagnostic only and does not earn credit without a correct
verifier-side target. A non-zero score requires a complete verifier run and all
required metrics to be finite.
