# MDN Component-Balance Regularization

Choose the strength of a component-usage balance penalty for a fixed MDN.

Edit `mdn-density/solution/component_balance.py`. `surface_config()` must return
a JSON literal with exactly one key, `component_balance_weight`. The finite
value must be in `[0, 1]`. During training, this coefficient weights the KL
divergence between the minibatch-average component usage and a uniform prior.
The held-out metric remains unregularized mixture NLL. Component count, trunk,
variance head, optimizer, data, and training budget remain fixed.

Evaluation uses 20,000 train and 20,000 test examples for 4,000 updates and
reports exact held-out mixture NLL. Lower NLL is preferred.
