# Mamba Post-Convolution Activation

Set the literal `activation` field returned by `surface_config()` in
`mamba/solution/conv_act.py` to `identity`, `relu`, or `silu`. Trusted verifier
code applies it to the fixed depthwise causal-convolution output. Every other
model surface is fixed, including the two-layer depth from the paper protocol.

The evaluator follows the selective-copying protocol in Mamba Appendix E.1:
total sequence length 4096, 16 memorized tokens, vocabulary 16, two D=64 layers,
batch 64, Adam at constant learning rate 1e-4, and 400,000 steps. The complete
CUDA training and evaluation proof is required. Token copy accuracy is the metric.
