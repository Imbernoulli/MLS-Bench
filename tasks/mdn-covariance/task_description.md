# MDN Covariance Structure

Choose the covariance structure of each component in a fixed bivariate MDN on the fixed conditional-density dataset.

Edit `mdn-density/solution/covariance.py`. `surface_config()` must return a JSON
literal with exactly one key, `covariance`. Valid values are `diag` and `full`.
Component count, trunk, initialization, and optimizer are fixed.

Evaluation uses 20,000 train and 20,000 test examples for 4,000 updates and
reports exact bivariate mixture NLL. Lower NLL is preferred.
