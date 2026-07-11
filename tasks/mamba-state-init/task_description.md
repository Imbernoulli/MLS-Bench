# Mamba State-Matrix Spectrum Initialization

Set the literal `scheme` field returned by `surface_config()` in
`mamba/solution/state_init.py` to `constant_rate` or `s4d_spectrum`. The fixed
state transform is `A = -exp(A_log)`: `constant_rate` initializes every entry to
`A=-1`, while `s4d_spectrum` initializes the real spectrum `-(1, ..., N)`.
Trusted verifier code protects Delta initialization and all other model state.

The evaluator follows the selective-copying protocol in Mamba Appendix E.1:
total sequence length 4096, 16 memorized tokens, vocabulary 16, two D=64 layers,
batch 64, Adam at constant learning rate 1e-4, and 400,000 steps. The complete
CUDA training and evaluation proof is required. Token copy accuracy is the metric.
