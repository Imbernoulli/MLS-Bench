# Differentially Private SGD: Privacy-Utility Optimization

## Research Question
Design an improved DP-SGD variant that achieves better privacy-utility tradeoff — higher test accuracy under the same `(epsilon, delta)`-differential privacy budget.

## Background
Differentially Private Stochastic Gradient Descent (DP-SGD) was introduced in Abadi et al., "Deep Learning with Differential Privacy" (CCS 2016; arXiv:1607.00133). The mechanism has two steps: (1) clip each per-sample gradient to a fixed `L2`-norm `C`, and (2) add Gaussian noise of scale `σC` to the aggregated gradient before the optimizer step. The noise multiplier `σ` is calibrated to the desired `(ε, δ)` budget via the moments accountant or RDP/PRV accountants.

A constant clipping threshold and constant noise schedule are suboptimal: gradient magnitudes evolve during training, so a fixed threshold either over-clips (losing useful signal) or under-clips (adding excess noise relative to the post-clip norm), and uniform noise allocation ignores varying gradient informativeness across stages. Recent work explores adaptive clipping (Andrew et al., NeurIPS 2021; arXiv:1905.03871), automatic per-sample clipping (Bu et al., "Automatic Clipping", NeurIPS 2023), and noise-decay schedules.

## Task
Modify the `DPMechanism` class in `custom_dpsgd.py`. Your mechanism receives per-sample gradients and decides how each one is clipped (per-sample multipliers and the clipping bound `C_t`) and the noise multiplier `σ_t` of each step. You control gradient clipping strategy, noise calibration, and any per-step adaptation. The FIXED harness applies your multipliers, checks that every clipped per-sample gradient has `L2`-norm at most `C_t`, averages over the batch, adds Gaussian noise of std `σ_t·C_t/B`, and accounts the privacy spent from the `σ_t` it applied.

## Interface
```python
class DPMechanism:
    def __init__(self, max_grad_norm, noise_multiplier, n_params,
                 dataset_size, batch_size, epochs, target_epsilon, target_delta):
        ...

    def clip(self, per_sample_grads, step, epoch) -> tuple[Tensor | list[Tensor], float]:
        # per_sample_grads: copy of the list of tensors [B, *param_shape]
        # Returns (scale, clip_norm): per-sample multipliers, a [B] tensor
        # (or a list of [B] tensors, one per parameter), and the L2 bound C_t
        # that every scaled per-sample gradient satisfies
        ...

    def get_noise_multiplier(self, step, epoch) -> float:
        # Returns this step's noise multiplier sigma_t (called after clip);
        # the harness adds N(0, (sigma_t * C_t / B)^2) noise and accounts it
        ...
```

## Constraints
- The total privacy budget `(target_epsilon, target_delta)` is FIXED and checked externally: the harness composes the per-step `σ_t` it applied, and a run whose epsilon exceeds the target by more than 1% (the slack for the noise calibration's binary search), or whose clipped per-sample gradient exceeds its declared `C_t`, is aborted.
- The model architecture, data pipeline, optimizer, and training loop are FIXED; the harness aborts a run that hooks or monkeypatches the functions it relies on (see `_check_harness` in the file).
- Focus on algorithmic innovation in the DP mechanism: clipping strategies, noise schedules, gradient processing.
- Available imports: `torch`, `math`, `numpy` (via the FIXED section), `scipy.optimize`.

## Evaluation
Trained and evaluated on three datasets at `epsilon = 3.0`, `delta = 1e-5`:
- **MNIST** (28x28 grayscale digits, 10 classes)
- **Fashion-MNIST** (28x28 grayscale clothing, 10 classes)
- **CIFAR-10** (32x32 color images, 10 classes)

Metric: **test accuracy** (higher is better) under the same privacy budget. Privacy budget consumed is also recorded but not scored: every run is held to the same budget, and a run that exceeds it by more than the 1% calibration slack is aborted.

## Baselines (paper-cited reference implementations)
- **standard_dpsgd** — Abadi et al. (CCS 2016; arXiv:1607.00133): fixed `C` and constant `σ` calibrated up-front.
- **automatic_clipping** — Bu, Wang, Zha, and Karypis, "Automatic Clipping: Differentially Private Deep Learning Made Easier and Stronger" (NeurIPS 2023; arXiv:2206.07136): per-sample normalization removes the clipping-norm hyperparameter.
- **adaptive_clipping** — Andrew, Thakkar, McMahan, and Ramaswamy, "Differentially Private Learning with Adaptive Clipping" (NeurIPS 2021; arXiv:1905.03871): track an online private quantile of the per-sample norm.
- **noise_decay** — schedule the noise multiplier downward as training proceeds, accounting for the full schedule with the same target `(ε, δ)`.
