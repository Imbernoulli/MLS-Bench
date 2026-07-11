# Mamba Delta-Bias Initialization

Set the literal `scheme` field returned by `surface_config()` in
`mamba/solution/dt_init.py` to `too_large`, `too_small`, or `log_uniform_s4d`.
Trusted verifier code initializes only `dt_proj.bias` and `dt_const`; the state
matrix, modules, parameters, and other hooks are protected.

The evaluator follows the selective-copying protocol in Mamba Appendix E.1:
total sequence length 4096, 16 memorized tokens, vocabulary 16, two D=64 layers,
batch 64, Adam at constant learning rate 1e-4, and 400,000 steps. The complete
CUDA training and evaluation proof is required. Token copy accuracy is the metric.
