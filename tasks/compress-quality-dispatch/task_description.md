# Quality-band codec dispatch

Choose an official codec family separately for low, middle, and high quality-index bands.

Edit `compressai/solution/quality_policy.py` so `quality_policy()` returns a literal policy. The tuple is ordered `(qualities_1_to_3, qualities_4_to_6, qualities_7_to_8)`.
Every policy entry must be one of `factorized`, `hyperprior_scale`, or `meanscale`, selecting the pinned official CompressAI families `bmshj2018-factorized`, `bmshj2018-hyperprior`, and `mbt2018-mean`.

Verification evaluates the selected codec dispatch with real `compress()` and `decompress()` calls on all 24 Kodak images at qualities 1 through 8. The complete 192-case matrix is required, and scoring includes the full set plus all three fixed content strata. Invalid policies or incomplete output receive zero.
