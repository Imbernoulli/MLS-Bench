# Conditional Density Family

Select one conditional output family for a fixed training pipeline.

Edit `mdn-density/solution/density_bench.py`. `surface_config()` must return a
JSON literal with exactly one key, `density_family`. Valid values are `point`,
`single_gaussian`, and `mixture`. Each maps to a frozen model recipe.

The same choice is trained separately on three fixed conditional-density
evaluation datasets. Each uses 20,000 train and 20,000 test examples for 4,000
updates; all three held-out NLL values participate in the score. Lower NLL is
preferred.
