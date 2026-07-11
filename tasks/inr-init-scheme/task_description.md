# INR Signal Fitting: Initialization Scheme

## Objective

Investigate the coordinate-network parameter initialization policy inside the repository fixed signal-fitting pipeline. Modify only the declared editable file and select a design using the public contract and feedback from valid runs.

## Editable Surface

- File: `inr-signal-fitting/solution/init_scheme.py`
- Public symbol: `fit_inr` (task currently dropped)

This task is excluded because the measured candidates do not isolate one variable. Its callable is retained only for provenance and must not be shipped without a new fixed-frequency protocol.

This surface is not active while the task remains dropped.

## Evaluation

The fixed harness evaluates full-grid RGB reconstruction with the fixed signal data. It reports PSNR in dB; higher is better. Every configured evaluation contributes to the task score.

Do not modify the harness, scorer, data, scripts, or unrelated solution files.
