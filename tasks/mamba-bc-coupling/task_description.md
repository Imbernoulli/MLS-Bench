# Mamba Read/Write Coupling

Set the literal `coupling` field returned by `surface_config()` in
`mamba/solution/bc_coupling.py` to `tied`, `constant`, or `independent`. Trusted
verifier code constructs the read tensor. The constant option preserves one
state vector per inner channel; all non-coupling surfaces are fixed.

The evaluator follows the selective-copying protocol in Mamba Appendix E.1:
total sequence length 4096, 16 memorized tokens, vocabulary 16, two D=64 layers,
batch 64, Adam at constant learning rate 1e-4, and 400,000 steps. The complete
CUDA training and evaluation proof is required. Token copy accuracy is the metric.
