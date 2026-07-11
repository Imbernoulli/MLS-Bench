# INR Signal Fitting: RGB Jacobian Smoothness Regularization

## Objective

Investigate the weight of a per-channel RGB coordinate-Jacobian penalty in the
repository's fixed signal-fitting pipeline. The penalty is the mean squared
Frobenius norm of the three output-channel gradients with respect to the two
input coordinates; channels are differentiated separately and cannot cancel.

## Editable Surface

- File: `inr-signal-fitting/solution/jacobian_reg.py`
- Public symbol: `surface_config`

`surface_config()` must contain exactly one `return` statement whose value is a
finite JSON literal. The verifier parses this literal without executing agent
code, validates its exact keys, types, and ranges, then builds and trains the
rest of the pipeline. Function calls, side effects, decorators, arguments, and
extra statements are invalid.

Schema: `{"weight": <number>}` with `0 <= weight <= 10`.

The selected surface is active. A load error, training exception, malformed
return, missing output, wrong shape, or NaN/Inf invalidates the run; the harness
does not replace a failed implementation with another predictor.

## Evaluation

The fixed harness evaluates full-grid RGB reconstruction with the fixed signal
data. It reports PSNR in dB; higher is better. Every configured evaluation
contributes to the task score once task-specific anchors have been established.

Do not modify the harness, scorer, data, scripts, or unrelated solution files.
