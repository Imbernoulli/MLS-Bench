# Learned Image Compression: Entropy-Model Family

## Objective
Choose the entropy-model family used by a learned image codec. The comparison
uses the official CompressAI 1.2.8 model-zoo checkpoints, so it measures mature
rate-distortion behavior rather than the early optimization behavior of a small
codec trained inside the verifier.

## Interface
Edit `entropy_model` in `compressai/solution/entropy_model.py`:

```python
def entropy_model() -> str:
    return "factorized"
```

The result must be exactly one of `"factorized"`, `"hyperprior_scale"`, or
`"meanscale"`. They select the official `bmshj2018-factorized`,
`bmshj2018-hyperprior`, and `mbt2018-mean` MSE model families, respectively.
Malformed values and exceptions fail evaluation; no alternate model is used.
All three choices are complete, valid codecs: the factorized prior is the
simpler reference, the scale hyperprior is the community reference, and the
mean-scale model is the released extension. A valid weaker codec is scored from
its measured rate-distortion result; it is not a failure fallback. Invalid
selection or failed verification receives zero instead.

## Fixed Evaluation
The selected family is evaluated at all eight official quality levels. These
are the MSE checkpoints released by the CompressAI authors, trained on
Vimeo-90K rather than fitted inside this benchmark. Checkpoint files, their
official source URLs and hashes, the complete CompressAI source/native-extension
inventory, and all 24 Kodak image hashes are fixed in the evaluation image. No
fitting, downloading, package installation, archive extraction, or compilation
occurs during final verification.

For every quality and every full-resolution Kodak image, the evaluator calls
`compress()` and `decompress()`. Bitrate is the exact total serialized entropy
stream length in bits divided by the original image pixels. PSNR is computed on
the decoded RGB tensor after removing only the padding required by the codec.

At each quality, `utility = mean_PSNR - 12 * aggregate_bpp`. The metric for a
setting is the mean utility across quality levels 1 through 8. The complete
Kodak-24 aggregate and the low-, medium-, and high-texture eight-image strata
are all required and all participate in the task score. Incomplete case grids,
wrong inventories, failed commands, non-finite values, or missing completion
proof receive zero.
