# Optimization Parity

## Research Question
Can you improve a fixed two-layer MLP's ability to learn sparse parity by designing only its initialization, training dataset, and AdamW hyperparameters?

## Background
The k-sparse parity problem maps a binary vector `x ∈ {0, 1}^N` to `y = (sum_{i in S} x_i) mod 2` for an unknown subset `S` of size `k = 8`. It is statistically easy but computationally hard (SQ-hard in `n^Ω(k)`), and it has become a canonical "feature-learning" benchmark. Barak, Edelman, Goel, Kakade, Malach, and Zhang, "Hidden Progress in Deep Learning: SGD Learns Parities Near the Computational Limit" (NeurIPS 2022; arXiv:2207.08799), show that vanilla SGD on a wide MLP undergoes a phase transition: the loss curve looks flat for a long time while a Fourier gap in the population gradient slowly amplifies, and only then does test accuracy jump.

In this benchmark the model architecture, optimizer family, batch size, training loop, and evaluation protocol are fixed. Your scientific freedom is in **initialization**, **training data construction**, and **AdamW hyperparameters** — the three knobs that prior work suggests can move the phase transition forward by orders of magnitude.

## What You Can Modify
Edit the scaffold file `pytorch-examples/optimization_parity/custom_strategy.py` only inside the editable block containing:

1. `init_model(model, config)`
2. `make_dataset(x_pool, y_pool, config)`
3. `get_optimizer_config(config)`

The benchmark is evaluated on three configurations: `(N=32, K=8)`, `(N=50, K=8)`, and `(N=64, K=8)`, all with `W = 512`.

## Fixed Setup
- Task: `y = (sum_{i in S} x_i) mod 2` for a hidden secret subset `S` of size `K = 8`.
- Inputs: binary vectors `x in {0, 1}^N`.
- Model: `Linear(N, W) -> ReLU -> Linear(W, 1) -> Sigmoid` with `W = 512`.
- Optimizer type: `AdamW`.
- Loss: binary cross-entropy.
- Batch size: 128.
- Training budget: up to 100,000 steps, reshuffling every epoch.
- Evaluation: 10 hidden secrets × 10 random epoch-orderings per secret = 100 runs; report mean held-out test accuracy.

## Interface Notes
- The harness supplies a large pool of labeled binary training examples `(x_pool, y_pool)` drawn from the task distribution. The hidden secret subset `S` is never exposed to your code; recovering the relevant features from `(x_pool, y_pool)` is the learning problem.
- `init_model(...)` must not depend on the hidden secret.
- `make_dataset(x_pool, y_pool, config)` selects/transforms the provided labeled pool and must return either `(x, y)` or `{"x": x, "y": y}`.
- `x` must have shape `[num_examples, N]` with binary values only.
- `y` must have shape `[num_examples]` (or `[num_examples, 1]`) with binary labels.
- Training dataset size must stay `<= 12_800_000` examples.
- `get_optimizer_config(...)` must return `lr`, `wd`, `beta1`, and `beta2`.

## Metric
The leaderboard metric is `test_accuracy` (also emitted as `score`), the mean test accuracy across all 100 training runs. Higher is better.

## Baselines (variants of the reference setup)
- **default** — single-pass training over freshly sampled examples with default AdamW settings (`lr = 1e-3`, `wd = 1e-2`, `(beta1, beta2) = (0.9, 0.999)`), the baseline analysed by Barak et al. (NeurIPS 2022; arXiv:2207.08799).
- **multi_epoch** — same configuration as `default` but iterating over a smaller fixed dataset for many epochs to test the impact of finite data and reshuffling.
- **nowd** — same as `default` but with `wd = 0`, isolating the role of weight decay during the slow-amplification phase identified in the paper.
