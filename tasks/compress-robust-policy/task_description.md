# Content-robust codec policy

Choose a family per content stratum; scoring requires performance on every stratum and the full set.

Edit `compressai/solution/robust_policy.py` so `robust_policy()` returns a literal policy. The returned tuple is ordered `(low_texture, mid_texture, high_texture)`.
Every policy entry must be one of `factorized`, `hyperprior_scale`, or `meanscale`, selecting the pinned official CompressAI families `bmshj2018-factorized`, `bmshj2018-hyperprior`, and `mbt2018-mean`.

Verification evaluates the selected codec dispatch with real `compress()` and `decompress()` calls on all 24 Kodak images at qualities 1 through 8. The complete 192-case matrix is required, and scoring includes the full set plus all three fixed content strata. Invalid policies or incomplete output receive zero.
