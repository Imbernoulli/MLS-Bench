# MDN Trunk Activation

Choose the trunk activation for a fixed six-component Mixture Density Network
trained on the fixed conditional-density dataset.

Edit `mdn-density/solution/activation.py`. `surface_config()` must return a JSON
literal with exactly one key, `activation`. Valid values are `tanh`, `relu`,
`gelu`, `sigmoid`, and `elu`. Width, depth, head, and optimizer are fixed.

Evaluation uses 20,000 train and 20,000 test examples for 4,000 updates and
reports exact held-out mixture NLL. Lower NLL is preferred.
