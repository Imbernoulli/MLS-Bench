# MDN Trunk Width

Choose the hidden width of a fixed-depth MDN trunk trained on the fixed conditional-density dataset.

Edit `mdn-density/solution/network_width.py`. `surface_config()` must return a
JSON literal with exactly one key, `network_width`. The integer value must be in
`[2, 128]`. Activation, depth, output head, component count, and optimizer are
fixed. The verifier also reports the resulting parameter count.

Evaluation uses 20,000 train and 20,000 test examples for 4,000 updates and
reports exact held-out mixture NLL. Lower NLL is preferred.
