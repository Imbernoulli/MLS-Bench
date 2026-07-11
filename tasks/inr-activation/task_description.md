# INR Signal Fitting: Representation Family

## Objective

Investigate the coordinate-network representation family inside the repository fixed signal-fitting pipeline. The categorical choices bundle their architecture-specific activation, encoding, and initialization; this is a family comparison, not an isolated activation ablation. Modify only the declared editable file and select a design using the public contract and feedback from valid runs.

## Editable Surface

- File: `inr-signal-fitting/solution/activation.py`
- Public symbol: `surface_config`

`surface_config()` must contain exactly one `return` statement whose value is a finite JSON literal. The verifier parses this literal without executing agent code, validates its exact keys, types, and ranges, then builds and trains the rest of the pipeline. Function calls, side effects, decorators, arguments, and extra statements are invalid.

Schema: `{"family": <string>}` where `family` is `relu`, `fourier`, or `siren`.



The selected surface is active. A load error, training exception, malformed return, missing output, wrong shape, or NaN/Inf invalidates the run; the harness does not replace a failed implementation with another predictor.

## Evaluation

The fixed harness evaluates full-grid RGB reconstruction with the fixed signal data. It reports PSNR in dB; higher is better. Every configured evaluation contributes to the task score.

Do not modify the harness, scorer, data, scripts, or unrelated solution files.
