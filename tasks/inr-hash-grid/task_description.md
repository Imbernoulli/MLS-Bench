# INR Signal Fitting: Hash Grid Configuration

## Objective

Investigate the multiresolution hash-grid configuration inside the repository fixed signal-fitting pipeline. Modify only the declared editable file and select a design using the public contract and feedback from valid runs.

## Editable Surface

- File: `inr-signal-fitting/solution/hash_grid.py`
- Public symbol: `surface_config`

`surface_config()` must contain exactly one `return` statement whose value is a finite JSON literal. The verifier parses this literal without executing agent code, validates its exact keys, types, and ranges, then builds and trains the rest of the pipeline. Function calls, side effects, decorators, arguments, and extra statements are invalid.

Schema: `{"n_levels": <integer>, "base_res": <integer>, "finest_res": <integer>}` with `1 <= n_levels <= 16`, `2 <= base_res <= 256`, and `base_res <= finest_res <= 512`; a one-level grid requires equal base and finest resolutions.



The selected surface is active. A load error, training exception, malformed return, missing output, wrong shape, or NaN/Inf invalidates the run; the harness does not replace a failed implementation with another predictor.

## Evaluation

The fixed harness evaluates full-grid RGB reconstruction with the fixed signal data. It reports PSNR in dB; higher is better. Every configured evaluation contributes to the task score.

Do not modify the harness, scorer, data, scripts, or unrelated solution files.
