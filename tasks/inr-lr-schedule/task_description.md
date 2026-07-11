# INR Signal Fitting: Learning Rate Schedule

## Objective

Investigate the optimizer learning-rate and schedule configuration inside the repository fixed signal-fitting pipeline. Modify only the declared editable file and select a design using the public contract and feedback from valid runs.

## Editable Surface

- File: `inr-signal-fitting/solution/lr_schedule.py`
- Public symbol: `surface_config`

`surface_config()` must contain exactly one `return` statement whose value is a finite JSON literal. The verifier parses this literal without executing agent code, validates its exact keys, types, and ranges, then builds and trains the rest of the pipeline. Function calls, side effects, decorators, arguments, and extra statements are invalid.

Schema: `{"lr": <number>, "schedule": <string>}` with `0 < lr <= 0.5` and `schedule` equal to `constant` or `cosine`.



The selected surface is active. A load error, training exception, malformed return, missing output, wrong shape, or NaN/Inf invalidates the run; the harness does not replace a failed implementation with another predictor.

## Evaluation

The fixed harness evaluates full-grid RGB reconstruction with the fixed signal data. It reports PSNR in dB; higher is better. Every configured evaluation contributes to the task score.

Do not modify the harness, scorer, data, scripts, or unrelated solution files.
