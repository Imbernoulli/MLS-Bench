# INR Signal Fitting: Fourier Frequency

## Objective

Investigate the frequency scale of the fixed-size Fourier encoding inside the repository fixed signal-fitting pipeline. Modify only the declared editable file and select a design using the public contract and feedback from valid runs.

## Editable Surface

- File: `inr-signal-fitting/solution/frequency.py`
- Public symbol: `surface_config`

`surface_config()` must contain exactly one `return` statement whose value is a finite JSON literal. The verifier parses this literal without executing agent code, validates its exact keys, types, and ranges, then builds and trains the rest of the pipeline. Function calls, side effects, decorators, arguments, and extra statements are invalid.

Schema: `{"sigma": <number>}` with `0 < sigma <= 100`.



The selected surface is active. A load error, training exception, malformed return, missing output, wrong shape, or NaN/Inf invalidates the run; the harness does not replace a failed implementation with another predictor.

## Evaluation

The fixed harness reports full-grid RGB reconstruction PSNR in dB; higher is
better. A non-zero score requires a complete verifier run and all required
metrics to be finite.

Do not modify the harness, scorer, data, scripts, or unrelated solution files.
