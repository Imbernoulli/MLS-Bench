# Gradient Compression for Communication-Efficient Distributed Training

## Research Question
Design a gradient compression operator that reduces communication cost in distributed training while maintaining convergence quality (test accuracy).

## Background
In distributed data-parallel training, gradient communication is often the bottleneck. Workers compute local gradients, which must be aggregated (e.g., via all-reduce) before the optimizer step. Gradient compression reduces the volume of data communicated by applying lossy compression to gradients before transmission.

Three main families of compression exist:
- **Sparsification**: keep only a subset of gradient elements (e.g., TopK selects the largest magnitudes; Stich, Cordonnier, and Jaggi, "Sparsified SGD with Memory", NeurIPS 2018).
- **Quantization**: reduce the precision of gradient values (e.g., QSGD uses stochastic rounding to discrete levels).
- **Low-rank approximation**: approximate gradient matrices with low-rank factors (e.g., PowerSGD).

A key challenge is that naive compression introduces bias or variance that degrades convergence. Error feedback — accumulating compression residuals locally and adding them to the next gradient — is a widely used correction (Karimireddy, Rebjock, Stich, and Jaggi, "Error Feedback Fixes SignSGD and Other Gradient Compression Schemes", ICML 2019; arXiv:1901.09847).

## Task
Modify the `Compressor` class in `custom_compressor.py`. Your compressor must implement:
- `__init__(self, compress_ratio)`: initialize with a target compression ratio (`0.01` = 100x compression).
- `compress(self, tensor, name)`: compress a gradient tensor, returning `(compressed_tensors, ctx)`.
- `decompress(self, compressed_tensors, ctx)`: reconstruct the gradient.

The compressor may maintain internal state (e.g., error feedback residuals) across calls. The `name` parameter identifies parameters for per-parameter state tracking.

## Interface
```python
class Compressor:
    def __init__(self, compress_ratio=0.01): ...
    def compress(self, tensor, name) -> (list[Tensor], ctx): ...
    def decompress(self, compressed_tensors, ctx) -> Tensor: ...
```
- `compress_ratio`: fraction of gradient elements/information to retain (`0.01` = keep 1%).
- `compressed_tensors`: list of tensors that would be communicated over the network.
- `ctx`: local context (not communicated) needed for decompression.
- The decompressed tensor must have the same shape as the original input.

## Evaluation
Trained and evaluated on three settings with 100x compression (`compress_ratio = 0.01`):
- **ResNet-20 / CIFAR-10** (~0.27M params): small model, standard benchmark.
- **VGG-11-BN / CIFAR-100** (~9.8M params): larger model, harder 100-class problem.
- **ResNet-56 / CIFAR-10** (~0.85M params): deeper model, tests scalability.

Metric: **best test accuracy** (higher is better). All settings use SGD with momentum, cosine LR schedule, and 200 training epochs.

## Baselines (paper-cited reference implementations)
- **topk_ef** — Top-K sparsification with error feedback (Stich et al., "Sparsified SGD with Memory", NeurIPS 2018; Karimireddy et al., "Error Feedback Fixes SignSGD and Other Gradient Compression Schemes", ICML 2019; arXiv:1901.09847). Keeps the `k = compress_ratio * d` largest-magnitude entries.
- **qsgd** — Quantized SGD with stochastic uniform quantization (Alistarh, Grubic, Li, Tomioka, and Vojnovic, "QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding", NeurIPS 2017; arXiv:1610.02132).
- **signsgd** — Sign-only gradient compression (Bernstein, Wang, Azizzadenesheli, and Anandkumar, "signSGD: Compressed Optimisation for Non-Convex Problems", ICML 2018; arXiv:1802.04434), typically combined with majority-vote aggregation.

A reference low-rank method (Vogels, Karimireddy, and Jaggi, "PowerSGD: Practical Low-Rank Gradient Compression for Distributed Optimization", NeurIPS 2019; arXiv:1905.13727) is a useful design point even though it is not run as a baseline here.
