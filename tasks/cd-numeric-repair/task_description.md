# Answer-Region Token Budget

## Research Question
Study how the constrained answer region's token budget affects dead ends and
numeric accuracy for a frozen instruction model. The prompt, trigger, accepted
numeric language, and free-reasoning budget remain fixed. Evaluation inputs and
targets are fixed; targets are available only during verification.

## Implementation Contract
Edit `constrained-decoding-lab/solution/decoder_repair.py` and implement:

```python
def build_decoder(question, tok):
    ...
```

Return `common.DecodeSpec` with the fixed full numeric language and an explicit
positive `max_answer_tokens`. Invalid or budget-exhausted outputs are measured
as produced. Missing fields, invalid types, non-finite model outputs, or runtime
exceptions fail verification; the harness does not substitute another decoder.

## Evaluation
The frozen model runs deterministically. Final assessment measures whether each
committed answer is both structurally valid and correct. Structural validity
alone is diagnostic and does not earn credit without a correct verifier target.
