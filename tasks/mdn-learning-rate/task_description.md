# MDN Learning Rate

Choose the Adam learning rate for a fixed five-component MDN trained on the fixed conditional-density dataset under a fixed step budget.

Edit `mdn-density/solution/learning_rate.py`. `surface_config()` must return a
JSON literal with exactly one key, `learning_rate`. The value must be finite and
in `[1e-5, 1e-1]`. Model, batch size, and optimizer family are fixed.

Evaluation uses 20,000 train and 20,000 test examples for 4,000 updates and
reports exact held-out mixture NLL. Lower NLL is preferred.
