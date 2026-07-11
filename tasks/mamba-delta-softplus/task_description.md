# Mamba Delta Finalization

Set the literal `activation` field returned by `surface_config()` in
`mamba/solution/delta_softplus.py` to `identity`, `relu`, or `softplus`. Trusted
verifier code adds the same fixed Delta bias before applying the selected rule,
so the three choices differ only in the finalization function.

The evaluator follows the selective-copying protocol in Mamba Appendix E.1:
total sequence length 4096, 16 memorized tokens, vocabulary 16, two D=64 layers,
batch 64, Adam at constant learning rate 1e-4, and 400,000 steps. The complete
CUDA training and evaluation proof is required. Token copy accuracy is the metric.
