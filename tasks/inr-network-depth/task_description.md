# INR Signal Fitting: Network Depth

## Objective

Investigate the number of hidden layers in the fixed-width Fourier-feature network. Modify only the declared editable file and select a configuration using feedback from valid runs.

## Editable Surface

- File: `inr-signal-fitting/solution/depth.py`
- Public symbol: `surface_config`

`surface_config()` must contain exactly one `return` statement whose value is a finite JSON literal matching this contract. The verifier parses the literal without executing agent code; calls, side effects, decorators, arguments, and extra statements are invalid.

Schema:

`{"n_layers": <integer>}` with `1 <= n_layers <= 12`.

The fixed verifier validates the value, then builds and trains the rest of the pipeline. A load error, exception, malformed configuration, missing output, wrong shape, or NaN/Inf invalidates the run. No substitute predictor or score is emitted after a verifier failure.

## Evaluation

The fixed verifier evaluates full-grid RGB reconstruction with the fixed signal data. It reports PSNR in dB; higher is better. Every configured evaluation contributes to the task score.

Do not modify the verifier, scorer, data, scripts, or unrelated solution files.
