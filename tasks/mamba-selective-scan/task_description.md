# Mamba Selective Parameterization

Set the literal `mode` field returned by `surface_config()` in
`mamba/solution/selective_param.py` to `lti`, `bc_only`, or `selective`. Trusted
verifier code constructs `(dt, B, C, delta_bias)` for the fixed CUDA scan. The
convolution, state dynamics, gate, skip, normalization, and residual path are fixed.

The evaluator follows the selective-copying protocol in Mamba Appendix E.1:
total sequence length 4096, 16 memorized tokens, vocabulary 16, two D=64 layers,
batch 64, Adam at constant learning rate 1e-4, and 400,000 steps. The complete
CUDA training and evaluation proof is required. Token copy accuracy is the metric.
