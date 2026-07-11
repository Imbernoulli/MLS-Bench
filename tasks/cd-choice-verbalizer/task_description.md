# Verbalizer Design for Forced-Choice Decoding

## Research Question
Study how literal class verbalizers affect a frozen instruction model under a
complete, explicit one-to-one mapping back to the canonical label set. The task measures the fraction of
examples whose committed answer is both structurally valid and correct. Evaluation
inputs and targets are fixed; targets are available only during verification.

## Implementation Contract
Edit `constrained-decoding-lab/solution/decoder_choice_verbalizer.py` and implement:

```python
def build_decoder(text, labels, tok):
    ...
```

Return `common.DecodeSpec` with a prompt, a non-empty list of four unique choice
strings, and `choice_labels`, a same-length one-to-one mapping onto the supplied
canonical `labels`. This permits aliases or compact code verbalizers without
making them automatically wrong. The trusted evaluator maps the committed
choice back to a canonical label and compares it with verifier-only targets.
Missing fields, incomplete or duplicate mappings, invalid types, non-finite model outputs, or runtime
exceptions fail verification; the harness does not substitute another decoder.

## Evaluation
The frozen model runs deterministically. Final assessment measures whether each
committed answer is both structurally valid and correct. Structural validity alone
is diagnostic and does not earn credit without a correct verifier-side target.
