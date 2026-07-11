# Entropy-stream-budget codec choice

Choose one official codec family for the complete matrix under the entropy-stream constraint.

Edit `compressai/solution/stream_budget_policy.py` so `stream_budget_policy()` returns a literal policy. Return one family string.
Every policy entry must be one of `factorized`, `hyperprior_scale`, or `meanscale`, selecting the pinned official CompressAI families `bmshj2018-factorized`, `bmshj2018-hyperprior`, and `mbt2018-mean`.

Verification evaluates the selected codec dispatch with real `compress()` and `decompress()` calls on all 24 Kodak images at qualities 1 through 8. The complete 192-case matrix is required, and scoring includes the full set plus all three fixed content strata. Invalid policies or incomplete output receive zero.
