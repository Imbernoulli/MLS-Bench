# MDN Variance Initialization

Choose the initial component standard deviation for a fixed MDN trained on the fixed conditional-density dataset.

Edit `mdn-density/solution/initialization.py`. `surface_config()` must return a
JSON literal with exactly one key, `initial_sigma`. The value must be finite and
in `[1e-3, 10]`. Architecture, component count, variance formula, and optimizer
are fixed.

Evaluation uses 20,000 train and 20,000 test examples for 4,000 updates and
reports exact held-out mixture NLL. Lower NLL is preferred.
