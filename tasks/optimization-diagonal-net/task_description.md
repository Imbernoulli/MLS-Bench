# Optimizer Design for Diagonal-Net Sparse Recovery

## Research Question
Can you design an optimizer that recovers a sparse linear predictor from fewer training samples when the model uses a diagonal-net parameterization with noisy labels?

## Background
The diagonal-net reparameterizes a linear model as `w = u^2 - v^2` (element-wise), where `u, v ∈ R^d` are the trainable parameters. Despite being equivalent to a linear predictor, the squared parameterization creates a non-convex loss landscape whose geometry interacts with the optimizer's implicit bias. Classical and recent results (e.g. Pesme, Pillaud-Vivien, and Flammarion, "Implicit Bias of SGD for Diagonal Linear Networks: a Provable Benefit of Stochasticity", NeurIPS 2021; arXiv:2106.09524) show that gradient-based methods on this parameterization can achieve implicit sparse regularization — the optimizer's dynamics naturally favour sparse solutions without explicit L1 penalties.

The benchmark uses **PyTorch with autograd** for gradient computation. Each training step adds fresh Rademacher noise `ζ_t ∈ {-delta, +delta}` to the labels before computing the loss, simulating stochastic perturbations. Test evaluation always uses clean (noise-free) labels.

The critical quantity is the **sample complexity of recovery**: how many training examples `n` does the optimizer need to reliably recover a `k`-sparse ground truth in `R^d`? Different optimizers induce different implicit biases, leading to dramatically different sample requirements.

## Task
Modify the three functions in `RAIN/opt_diagonal_net/custom_optimizer.py` (inside the editable block) to implement a novel or improved optimizer:

1. `get_hyperparameters(dim, sparsity, delta, alpha_init)` — return optimizer configuration.
2. `init_state(u, v, hyperparameters)` — initialise optimizer state.
3. `step(u, v, grad_u, grad_v, state, hyperparameters)` — perform one update step.

The default template implements vanilla gradient descent. Your goal is to achieve successful recovery (test MSE < 1.0) with fewer training samples across all evaluation settings.

## Interface
- `u`, `v`: parameter vectors of shape `(d,)` as `torch.Tensor` (float64), initialised as `alpha/sqrt(2d) * ones(d)`, where `alpha` is the setting's `alpha_init` (passed to `get_hyperparameters`) and VARIES ACROSS SETTINGS.
- `grad_u`, `grad_v`: full-batch MSE gradients w.r.t. `u` and `v` (computed by PyTorch autograd).
- `state`: mutable dict for optimizer internal state (momentum buffers, accumulators, etc.).
- `hyperparameters`: dict returned by `get_hyperparameters`.
- `step()` must return `(u_new, v_new, state_new)` as a tuple of `torch.Tensor` and dict.
- All operations should use `torch` (not numpy); the benchmark provides gradients via autograd.
- The `delta` parameter controls the magnitude of Rademacher noise added to training labels each step.

### Training loop (executed by the benchmark)
```python
model.zero_grad()
noise = delta * (2 * torch.randint(0, 2, y_train.shape) - 1).float()
y_noisy = y_train + noise
loss = 0.5 * torch.mean((model(X_train) - y_noisy) ** 2)
loss.backward()
with torch.no_grad():
    u_new, v_new, state = step(u, v, grad_u, grad_v, state, hparams)
    model.u.data.copy_(u_new)
    model.v.data.copy_(v_new)
```

## Evaluation
Settings exercised by the harness include:
- **d200_k20_a1e3**: d=200, k=20, alpha=1e-3, delta=0.5.
- **d500_k10_a1e3**: d=500, k=10, alpha=1e-3, delta=0.5.
- **d500_k10_a5e1**: d=500, k=10, alpha=0.5, delta=0.5 — identical (d, k) to the
  setting above, so the initialisation scale `alpha` is the ONLY difference.
  Small alpha trains in the rich regime (implicit bias toward sparse solutions);
  alpha=0.5 is close to the lazy / kernel regime, where the bias is instead
  ell-2-like. An optimizer tuned for one regime need not win in the other.
- A larger-scale setting at d=10000, k=50, alpha=1.0.

For each setting, the benchmark performs a coarse-to-fine search over training-set sizes `n ∈ {50, 75, ..., 1600}` (with the larger setting using a wider range) to find the smallest `n*` where recovery succeeds on at least 4 of 5 seeds. Recovery means test MSE < 1.0 at the time training stops.

**Metric**: `score = -log2(n*)` per setting (higher is better — fewer samples needed).

Training uses full-batch gradients (with noisy labels) and a shared stopping rule: training halts when both train and test MSE have plateaued (two-window comparison over 20,000 steps), or after 1,000,000 steps.

## Baselines (16 paper-default configurations)
- **SGD** (4 configs): lr ∈ {0.005, 0.01, 0.05, 0.1}.
- **AdaGrad** (4 configs): lr ∈ {0.005, 0.01, 0.05, 0.1}, eps=1e-6 (Duchi, Hazan, and Singer, JMLR 2011).
- **Adam without bias correction** (8 configs): lr ∈ {0.005, 0.01, 0.05, 0.1} × beta2 ∈ {0.95, 0.999}, beta1=0.9, eps=1e-6 (Kingma and Ba, "Adam", ICLR 2015; arXiv:1412.6980 — bias correction is intentionally omitted to study the raw adaptive geometry).

## Hints
- The diagonal-net parameterization `w = u^2 - v^2` naturally biases gradient descent toward sparse solutions when initialised near zero.
- Adaptive methods (Adam, AdaGrad) change the effective geometry of this bias — this can help or hurt.
- The initialisation `alpha/sqrt(2d) * ones(d)` means u=v at init, so w_hat=0 initially. `alpha` varies BY SETTING and is passed to `get_hyperparameters`, so hyperparameters may depend on it.
- The Rademacher noise (delta parameter) adds stochasticity to training — your optimizer should be robust to this.
- Consider how your optimizer interacts with the non-convex structure: coordinate-wise adaptivity, momentum, and learning rate scheduling all affect the sparsity bias.
