# Variance Reduction for Stochastic Optimization

## Research Question
Design an improved variance reduction strategy for stochastic gradient descent on finite-sum optimization problems. Your method should accelerate convergence compared to vanilla mini-batch SGD by reducing the variance of gradient estimates.

## Background
Many machine learning problems take the form of finite-sum optimization:

    min_x  F(x) = (1/n) * sum_{i=1}^{n} f_i(x)

Standard SGD uses a stochastic gradient from a random mini-batch, which has variance proportional to `1 / b` (where `b` is the batch size). Variance reduction methods use auxiliary information (snapshots, recursive corrections, momentum) to reduce this variance, enabling faster convergence — often achieving linear convergence rates for strongly convex problems where SGD only achieves sublinear rates.

Key methods in this area:
- **SVRG** — periodic full-gradient snapshot + control variate (Johnson and Zhang, "Accelerating Stochastic Gradient Descent using Predictive Variance Reduction", NeurIPS 2013).
- **SARAH** — recursive gradient correction (Nguyen, Liu, Scheinberg, and Takáč, "SARAH: A Novel Method for Machine Learning Problems Using Stochastic Recursive Gradient", ICML 2017; arXiv:1703.00102).
- **STORM** — momentum-based online variance reduction (Cutkosky and Orabona, "Momentum-Based Variance Reduction in Non-Convex SGD", NeurIPS 2019; arXiv:1905.10018).
- **STORM+** — fully adaptive STORM without smoothness/gradient-norm constants (Levy, Kavis, and Cevher, "STORM+: Fully Adaptive SGD with Recursive Momentum for Nonconvex Optimization", NeurIPS 2021; arXiv:2111.01040).
- **SPIDER / PAGE** — biased recursive estimators with optimal complexity for non-convex problems (Fang, Li, Lin, and Zhang, NeurIPS 2018; Li, Bao, Zhang, and Richtárik, ICML 2021).

## Task
Modify the `VarianceReductionOptimizer` class in `custom_vr.py` (inside the editable block). You must implement:

1. **`__init__(self, model, lr, l2_reg, loss_type, n_train, batch_size, device)`** — initialize any state needed for variance reduction (snapshot parameters, running gradient estimates, buffers, etc.).
2. **`train_one_epoch(self, X_train, y_train)`** — train for one epoch over the data, returning a dict with at least `'avg_loss'` (and optionally `'full_grad_count'` if you use full gradient computations).

The default implementation is vanilla mini-batch SGD. Your goal is to design a variance reduction mechanism that improves convergence.

## Interface

### Available helper functions (FIXED, use these for gradient computation):
```python
compute_full_gradient(model, X_train, y_train, loss_type, l2_reg, device)
# -> returns list of gradient tensors (one per parameter)

compute_stochastic_gradient(model, X_batch, y_batch, loss_type, l2_reg)
# -> returns list of gradient tensors for a mini-batch

compute_loss_on_batch(model, X_batch, y_batch, loss_type, l2_reg)
# -> returns scalar loss tensor
```

### Constraints
- You may call `compute_full_gradient` at most once per epoch.
- Parameter updates must use `p.data.add_(...)` or similar in-place operations.
- Must work across all problems with the same code.
- The learning rate (`self.lr`) and L2 regularization (`self.l2_reg`) are fixed.
- Do not modify the model architecture, loss function, or evaluation code.

## Evaluation
- **Problems**:
  - `logistic`: L2-regularized multinomial logistic regression on MNIST (convex, n=60K, 20 epochs).
  - `mlp`: 2-layer MLP on CIFAR-10 (non-convex, n=50K, 40 epochs).
  - `conditioned`: L2-regularized linear regression on synthetic ill-conditioned data (strongly convex, kappa=100, n=10K, 30 epochs).
- **Metrics**: `best_test_accuracy` and `final_test_accuracy` (logistic, mlp; higher is better) and `best_test_mse` / `final_test_mse` (conditioned; lower is better).
- All problems run in parallel with shared compute.

## Baselines (paper-cited reference implementations)
- **svrg** — Johnson and Zhang (NeurIPS 2013); paper-default outer-loop length `m = n / b` and a single full-gradient snapshot per epoch.
- **storm** — Cutkosky and Orabona (NeurIPS 2019; arXiv:1905.10018); paper-default momentum schedule `a_t = c / (k + t)^{2/3}` with the prescribed adaptive step size.
- **storm_plus** — Levy, Kavis, and Cevher (NeurIPS 2021; arXiv:2111.01040); paper-default fully adaptive step-size and momentum without prior knowledge of smoothness or gradient-norm bounds.
