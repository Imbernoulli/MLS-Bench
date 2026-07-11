# Mamba Block Normalization

Set the literal `normalization` field returned by `surface_config()` in
`mamba/solution/normalization.py` to `none`, `layer`, or `rms`. Trusted verifier
code constructs the module. LayerNorm and RMSNorm operate independently at each
token; no option combines statistics across examples or sequence positions.

The evaluator follows the selective-copying protocol in Mamba Appendix E.1:
total sequence length 4096, 16 memorized tokens, vocabulary 16, two D=64 layers,
batch 64, Adam at constant learning rate 1e-4, and 400,000 steps. The complete
CUDA training and evaluation proof is required. Token copy accuracy is the metric.
