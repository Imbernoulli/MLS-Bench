# MDN Component Count

Choose the Gaussian-mixture component count for a fixed MDN trained on the fixed conditional-density dataset.

Edit `mdn-density/solution/num_components.py`. `surface_config()` must return a
JSON literal with exactly one key, `num_components`. The integer value must be
in `[1, 16]`. Trunk, variance parameterization, initialization, and optimizer
are fixed.

Evaluation uses 20,000 train and 20,000 test examples for 4,000 updates and
reports exact held-out mixture NLL. Lower NLL is preferred.
