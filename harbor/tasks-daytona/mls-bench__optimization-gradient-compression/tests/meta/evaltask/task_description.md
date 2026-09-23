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
- `__init__(self, compress_ratio, param_numels, budget_bits)`: `param_numels` maps every parameter name to its number of entries; `budget_bits` is the communication budget of one training step.
- `compress(self, tensor, name)`: encode a gradient tensor into a packet (format below).

The compressor may maintain internal state (e.g., error feedback residuals) across calls. The `name` parameter identifies parameters for per-parameter state tracking. Decoding is fixed: the harness reconstructs each gradient from its packet alone, and only that reconstruction reaches the optimizer.

## Interface
```python
class Compressor:
    def __init__(self, compress_ratio, param_numels, budget_bits): ...
    def compress(self, tensor, name) -> packet: ...  # or a list of packets, whose decodings are summed
```
A packet is one of:
- `{"values": V, "bits": b}`: dense, one value per gradient entry;
- `{"values": V, "bits": b, "indices": I}`: sparse, `I` the distinct flat positions of the values `V`;
- `{"factors": (P, Q)}`: low rank, `P @ Q.T` viewed as `(shape[0], numel / shape[0])`.

Values are transmitted as float32. The harness charges each packet in bits: `len(V) * b`; plus 32 bits per distinct value when `b < 32` (the codebook, which may hold at most `2**b` values); plus the Elias-gamma code of the sorted gaps between sparse positions; plus 32 bits per low-rank factor entry. The helpers `packet_cost(packet, shape)` and `elias_gamma_bound(k, numel)` compute these charges.

**Budget.** One training step may transmit at most `budget_bits = compress_ratio × 32 × (total parameter entries)` bits over all parameters together, i.e. 100x less than the dense float32 gradient at `compress_ratio = 0.01`. A step over the budget, or a malformed packet (non-finite values, repeated or out-of-range indices, more distinct values than `2**b`), stops the run and the run is invalid.

## Evaluation
Trained and evaluated on three settings with 100x compression (`compress_ratio = 0.01`):
- **ResNet-20 / CIFAR-10** (~0.27M params): small model, standard benchmark.
- **VGG-11-BN / CIFAR-100** (~9.8M params): larger model, harder 100-class problem.
- **ResNet-56 / CIFAR-10** (~0.85M params): deeper model, tests scalability.

Metric: **best test accuracy** (higher is better). All settings use SGD with momentum, cosine LR schedule, and 200 training epochs.

## Baselines (paper-cited reference implementations, each sized to the budget)
- **topk_ef** — Top-K sparsification with error feedback (Stich et al., "Sparsified SGD with Memory", NeurIPS 2018; Karimireddy et al., "Error Feedback Fixes SignSGD and Other Gradient Compression Schemes", ICML 2019; arXiv:1901.09847). Sends the largest-magnitude entries as float32 values at Elias-gamma coded positions (about 0.65% of the entries at this budget).
- **qsgd** — Quantized SGD with stochastic uniform quantization (Alistarh, Grubic, Li, Tomioka, and Vojnovic, "QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding", NeurIPS 2017; arXiv:1610.02132), in the paper's Elias-coded sparse encoding with the largest number of levels that fits the budget.
- **signsgd** — Scaled sign compression with error feedback (Bernstein, Wang, Azizzadenesheli, and Anandkumar, "signSGD: Compressed Optimisation for Non-Convex Problems", ICML 2018; arXiv:1802.04434; Karimireddy et al. 2019). One bit for every entry is only 32x, so it sends the signs of the largest-magnitude entries the budget can carry (about 2.5%), scaled by their mean magnitude.

A reference low-rank method (Vogels, Karimireddy, and Jaggi, "PowerSGD: Practical Low-Rank Gradient Compression for Distributed Optimization", NeurIPS 2019; arXiv:1905.13727) is a useful design point even though it is not run as a baseline here.
