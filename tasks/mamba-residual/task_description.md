# Mamba Block Residual Rule

Set the literal `residual` field returned by `surface_config()` in
`mamba/solution/residual.py` to `none`, `scaled_add`, or `add`. Trusted verifier
code combines the residual stream and block output. `scaled_add` uses
`(hidden + block_out) / sqrt(2)` as a variance-preserving alternative; all block
internals and normalization are fixed.

The evaluator follows the selective-copying protocol in Mamba Appendix E.1:
total sequence length 4096, 16 memorized tokens, vocabulary 16, two D=64 layers,
batch 64, Adam at constant learning rate 1e-4, and 400,000 steps. The complete
CUDA training and evaluation proof is required. Token copy accuracy is the metric.
