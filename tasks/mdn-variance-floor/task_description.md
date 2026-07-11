# MDN Head-Side Variance Floor

Choose the standard-deviation floor applied by a fixed MDN head on the fixed conditional-density dataset.

Edit `mdn-density/solution/variance_floor.py`. `surface_config()` must return a
JSON literal with exactly one key, `variance_floor`. The value must be finite
and in `[0, 1]`. The evaluator retains an independent numerical guard at
`1e-3`; the selected head floor controls regularization above that guard.

Evaluation uses 20,000 train and 20,000 test examples for 4,000 updates and
reports exact held-out mixture NLL. Lower NLL is preferred.
