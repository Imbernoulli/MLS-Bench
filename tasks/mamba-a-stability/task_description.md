# Mamba State-Matrix Parameterization

Set the literal `transform` field returned by `surface_config()` in
`mamba/solution/a_stability.py` to `identity`, `neg_abs`, or `neg_exp`. Trusted
verifier code constructs the diagonal state matrix. Every other selective-SSM
surface is fixed.

The evaluator follows the selective-copying protocol in Mamba Appendix E.1:
total sequence length 4096, 16 memorized tokens, vocabulary 16, two D=64 layers,
batch 64, Adam at constant learning rate 1e-4, and 400,000 steps. The complete
CUDA training and evaluation proof is required. Token copy accuracy is the metric.
