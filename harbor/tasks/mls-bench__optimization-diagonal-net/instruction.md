# MLS-Bench: optimization-diagonal-net

# Optimizer Design for Diagonal-Net Sparse Recovery

## Research Question
Can you design an optimizer that recovers a sparse linear predictor from fewer training samples when the model uses a diagonal-net parameterization with noisy labels?

## Background
The diagonal-net reparameterizes a linear model as `w = u^2 - v^2` (element-wise), where `u, v ∈ R^d` are the trainable parameters. Despite being equivalent to a linear predictor, the squared parameterization creates a non-convex loss landscape whose geometry interacts with the optimizer's implicit bias. Classical and recent results (e.g. Pesme, Pillaud-Vivien, and Flammarion, "Implicit Bias of SGD for Diagonal Linear Networks: a Provable Benefit of Stochasticity", NeurIPS 2021; arXiv:2106.09524) show that gradient-based methods on this parameterization can achieve implicit sparse regularization — the optimizer's dynamics naturally favour sparse solutions without explicit L1 penalties.

The benchmark uses **PyTorch with autograd** for gradient computation. Each training step adds fresh Rademacher noise `ζ_t ∈ {-delta, +delta}` to the labels before computing the loss, simulating stochastic perturbations. Test evaluation always uses clean (noise-free) labels.

The critical quantity is the **sample complexity of recovery**: how many training examples `n` does the optimizer need to reliably recover a `k`-sparse ground truth in `R^d`? Different optimizers induce different implicit biases, leading to dramatically different sample requirements.

## Task
Modify the three functions in `RAIN/opt_diagonal_net/custom_optimizer.py` (inside the editable block) to implement a novel or improved optimizer:

1. `get_hyperparameters(dim, sparsity, noise_scale, delta)` — return optimizer configuration.
2. `init_state(u, v, hyperparameters)` — initialise optimizer state.
3. `step(u, v, grad_u, grad_v, state, hyperparameters)` — perform one update step.

The default template implements vanilla gradient descent. Your goal is to achieve successful recovery (test MSE < 1.0) with fewer training samples across all evaluation settings.

## Interface
- `u`, `v`: parameter vectors of shape `(d,)` as `torch.Tensor` (float64), initialised as `alpha/sqrt(2d) * ones(d)` with `alpha = 1e-3`.
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
- **d200_k5_s01**: d=200, k=5, sigma=0.1, delta=0.5.
- **d500_k10_s01**: d=500, k=10, sigma=0.1, delta=0.5.
- **d500_k10_s02**: d=500, k=10, sigma=0.2, delta=0.5.
- A larger-scale setting at d=10000, k=50.

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
- The initialisation `alpha/sqrt(2d) * ones(d)` with `alpha = 1e-3` means u=v at init, so w_hat=0 initially.
- The Rademacher noise (delta parameter) adds stochasticity to training — your optimizer should be robust to this.
- Consider how your optimizer interacts with the non-convex structure: coordinate-wise adaptivity, momentum, and learning rate scheduling all affect the sparsity bias.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/RAIN/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to score zero.

- `RAIN/opt_diagonal_net/custom_optimizer.py`
- editable lines **23–90**


Other files you may **read** for context (do not modify):
- `RAIN/opt_diagonal_net/fixed_benchmark.py`


## Readable Context


### `RAIN/opt_diagonal_net/custom_optimizer.py`  [EDITABLE — lines 23–90 only]

```python
     1: """Editable optimizer scaffold for the opt-diagonal-net MLS-Bench task.
     2: 
     3: Implement a custom optimizer for training a diagonal-net model to recover
     4: a sparse linear predictor.  You may edit the three functions below
     5: (get_hyperparameters, init_state, step) while the benchmark harness,
     6: data generation, model, stopping rule, and search protocol are fixed.
     7: """
     8: 
     9: from __future__ import annotations
    10: 
    11: from typing import Any
    12: 
    13: import torch
    14: 
    15: from fixed_benchmark import run_cli
    16: 
    17: 
    18: # =====================================================================
    19: # EDITABLE: get_hyperparameters, init_state, step  (lines 23 to 90)
    20: # =====================================================================
    21: 
    22: 
    23: def get_hyperparameters(
    24:     dim: int,
    25:     sparsity: int,
    26:     delta: float,
    27: ) -> dict[str, Any]:
    28:     """Return optimizer hyperparameters for this problem setting.
    29: 
    30:     Args:
    31:         dim: ambient dimension d.
    32:         sparsity: number of nonzero entries k in the ground truth.
    33:         delta: Rademacher noise magnitude (±delta) added to labels each step.
    34: 
    35:     Returns:
    36:         dict of hyperparameters used by init_state and step.
    37:     """
    38:     return {"lr": 0.01}
    39: 
    40: 
    41: def init_state(
    42:     u: torch.Tensor,
    43:     v: torch.Tensor,
    44:     hyperparameters: dict[str, Any],
    45: ) -> dict[str, Any]:
    46:     """Initialise optimizer state from the model parameters u, v.
    47: 
    48:     Args:
    49:         u: initial parameter vector u (shape (d,), float64).
    50:         v: initial parameter vector v (shape (d,), float64).
    51:         hyperparameters: dict from get_hyperparameters.
    52: 
    53:     Returns:
    54:         dict of optimizer state (passed to step and updated each iteration).
    55:     """
    56:     return {"t": 0}
    57: 
    58: 
    59: def step(
    60:     u: torch.Tensor,
    61:     v: torch.Tensor,
    62:     grad_u: torch.Tensor,
    63:     grad_v: torch.Tensor,
    64:     state: dict[str, Any],
    65:     hyperparameters: dict[str, Any],
    66: ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    67:     """Perform one optimizer step.
    68: 
    69:     Args:
    70:         u: current parameter u (shape (d,), float64).
    71:         v: current parameter v (shape (d,), float64).
    72:         grad_u: gradient of MSE loss w.r.t. u (shape (d,), float64).
    73:         grad_v: gradient of MSE loss w.r.t. v (shape (d,), float64).
    74:         state: mutable optimizer state from init_state / previous step.
    75:         hyperparameters: dict from get_hyperparameters.
    76: 
    77:     Returns:
    78:         (u_new, v_new, state_new) tuple of updated parameters and state.
    79:     """
    80:     lr = float(hyperparameters["lr"])
    81:     state["t"] = state.get("t", 0) + 1
    82:     return u - lr * grad_u, v - lr * grad_v, state
    83: 
    84: 
    85: 
    86: 
    87: 
    88: 
    89: 
    90: 
    91: if __name__ == "__main__":
    92:     run_cli(
    93:         get_hyperparameters=get_hyperparameters,
    94:         init_state=init_state,
    95:         step=step,
    96:     )
```




## How You Will Be Evaluated

After you finish, evaluation runs a fixed set of scripts and aggregates the
metrics they emit. These scripts are **not** in your workspace — you cannot
read or modify them. The labels below indicate what each evaluation tests:

- **d200_k5_s01** — wall-clock budget `04:00:00`, compute share `1.0`
- **d500_k10_s01** — wall-clock budget `04:00:00`, compute share `1.0`
- **d500_k10_s02** — wall-clock budget `04:00:00`, compute share `1.0`
- **d10000_k50** — wall-clock budget `08:00:00`, compute share `1.0`


Scoring uses the same `combined_score` aggregation as the MLS-Bench
leaderboard. Multiple seeds are averaged.



## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `sgd` baseline — editable region  [READ-ONLY — reference implementation]

In `RAIN/opt_diagonal_net/custom_optimizer.py`:

```python
Lines 23–52:
    20: # =====================================================================
    21: 
    22: 
    23: def get_hyperparameters(
    24:     dim: int,
    25:     sparsity: int,
    26:     delta: float,
    27: ) -> dict[str, Any]:
    28:     """SGD hyperparameters: lr=0.1."""
    29:     return {"lr": 0.1}
    30: 
    31: 
    32: def init_state(
    33:     u: torch.Tensor,
    34:     v: torch.Tensor,
    35:     hyperparameters: dict[str, Any],
    36: ) -> dict[str, Any]:
    37:     """SGD requires no additional state."""
    38:     return {"t": 0}
    39: 
    40: 
    41: def step(
    42:     u: torch.Tensor,
    43:     v: torch.Tensor,
    44:     grad_u: torch.Tensor,
    45:     grad_v: torch.Tensor,
    46:     state: dict[str, Any],
    47:     hyperparameters: dict[str, Any],
    48: ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    49:     """Vanilla gradient descent step."""
    50:     lr = float(hyperparameters["lr"])
    51:     state["t"] = state.get("t", 0) + 1
    52:     return u - lr * grad_u, v - lr * grad_v, state
    53: if __name__ == "__main__":
    54:     run_cli(
    55:         get_hyperparameters=get_hyperparameters,
```

### `adagrad` baseline — editable region  [READ-ONLY — reference implementation]

In `RAIN/opt_diagonal_net/custom_optimizer.py`:

```python
Lines 23–61:
    20: # =====================================================================
    21: 
    22: 
    23: def get_hyperparameters(
    24:     dim: int,
    25:     sparsity: int,
    26:     delta: float,
    27: ) -> dict[str, Any]:
    28:     """AdaGrad hyperparameters: lr=0.01, eps=1e-6."""
    29:     return {"lr": 0.01, "eps": 1e-6}
    30: 
    31: 
    32: def init_state(
    33:     u: torch.Tensor,
    34:     v: torch.Tensor,
    35:     hyperparameters: dict[str, Any],
    36: ) -> dict[str, Any]:
    37:     """AdaGrad state: accumulated squared gradients."""
    38:     d = u.shape[0]
    39:     return {
    40:         "t": 0,
    41:         "g_sum_u": torch.zeros(d, dtype=torch.float64),
    42:         "g_sum_v": torch.zeros(d, dtype=torch.float64),
    43:     }
    44: 
    45: 
    46: def step(
    47:     u: torch.Tensor,
    48:     v: torch.Tensor,
    49:     grad_u: torch.Tensor,
    50:     grad_v: torch.Tensor,
    51:     state: dict[str, Any],
    52:     hyperparameters: dict[str, Any],
    53: ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    54:     """AdaGrad update step."""
    55:     lr = float(hyperparameters["lr"])
    56:     eps = float(hyperparameters["eps"])
    57:     g_sum_u = state["g_sum_u"] + grad_u * grad_u
    58:     g_sum_v = state["g_sum_v"] + grad_v * grad_v
    59:     u_new = u - lr * grad_u / (torch.sqrt(g_sum_u) + eps)
    60:     v_new = v - lr * grad_v / (torch.sqrt(g_sum_v) + eps)
    61:     return u_new, v_new, {"t": state["t"] + 1, "g_sum_u": g_sum_u, "g_sum_v": g_sum_v}
    62: if __name__ == "__main__":
    63:     run_cli(
    64:         get_hyperparameters=get_hyperparameters,
```

### `adam` baseline — editable region  [READ-ONLY — reference implementation]

In `RAIN/opt_diagonal_net/custom_optimizer.py`:

```python
Lines 23–68:
    20: # =====================================================================
    21: 
    22: 
    23: def get_hyperparameters(
    24:     dim: int,
    25:     sparsity: int,
    26:     delta: float,
    27: ) -> dict[str, Any]:
    28:     """Adam (no bias correction) hyperparameters: lr=0.05, beta2=0.999."""
    29:     return {"lr": 0.05, "beta1": 0.9, "beta2": 0.999, "eps": 1e-6}
    30: 
    31: 
    32: def init_state(
    33:     u: torch.Tensor,
    34:     v: torch.Tensor,
    35:     hyperparameters: dict[str, Any],
    36: ) -> dict[str, Any]:
    37:     """Adam state: first and second moment estimates."""
    38:     d = u.shape[0]
    39:     return {
    40:         "t": 0,
    41:         "m_u": torch.zeros(d, dtype=torch.float64),
    42:         "s_u": torch.zeros(d, dtype=torch.float64),
    43:         "m_v": torch.zeros(d, dtype=torch.float64),
    44:         "s_v": torch.zeros(d, dtype=torch.float64),
    45:     }
    46: 
    47: 
    48: def step(
    49:     u: torch.Tensor,
    50:     v: torch.Tensor,
    51:     grad_u: torch.Tensor,
    52:     grad_v: torch.Tensor,
    53:     state: dict[str, Any],
    54:     hyperparameters: dict[str, Any],
    55: ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    56:     """Adam update step WITHOUT bias correction."""
    57:     lr = float(hyperparameters["lr"])
    58:     beta1 = float(hyperparameters["beta1"])
    59:     beta2 = float(hyperparameters["beta2"])
    60:     eps = float(hyperparameters["eps"])
    61:     t = state["t"] + 1
    62:     m_u = beta1 * state["m_u"] + (1.0 - beta1) * grad_u
    63:     s_u = beta2 * state["s_u"] + (1.0 - beta2) * grad_u * grad_u
    64:     u_new = u - lr * m_u / (torch.sqrt(s_u) + eps)
    65:     m_v = beta1 * state["m_v"] + (1.0 - beta1) * grad_v
    66:     s_v = beta2 * state["s_v"] + (1.0 - beta2) * grad_v * grad_v
    67:     v_new = v - lr * m_v / (torch.sqrt(s_v) + eps)
    68:     return u_new, v_new, {"t": t, "m_u": m_u, "s_u": s_u, "m_v": m_v, "s_v": s_v}
    69: if __name__ == "__main__":
    70:     run_cli(
    71:         get_hyperparameters=get_hyperparameters,
```

### `adam2` baseline — editable region  [READ-ONLY — reference implementation]

In `RAIN/opt_diagonal_net/custom_optimizer.py`:

```python
Lines 23–68:
    20: # =====================================================================
    21: 
    22: 
    23: def get_hyperparameters(
    24:     dim: int,
    25:     sparsity: int,
    26:     delta: float,
    27: ) -> dict[str, Any]:
    28:     """Adam (no bias correction) hyperparameters: lr=0.1, beta2=0.95."""
    29:     return {"lr": 0.1, "beta1": 0.9, "beta2": 0.95, "eps": 1e-6}
    30: 
    31: 
    32: def init_state(
    33:     u: torch.Tensor,
    34:     v: torch.Tensor,
    35:     hyperparameters: dict[str, Any],
    36: ) -> dict[str, Any]:
    37:     """Adam state: first and second moment estimates."""
    38:     d = u.shape[0]
    39:     return {
    40:         "t": 0,
    41:         "m_u": torch.zeros(d, dtype=torch.float64),
    42:         "s_u": torch.zeros(d, dtype=torch.float64),
    43:         "m_v": torch.zeros(d, dtype=torch.float64),
    44:         "s_v": torch.zeros(d, dtype=torch.float64),
    45:     }
    46: 
    47: 
    48: def step(
    49:     u: torch.Tensor,
    50:     v: torch.Tensor,
    51:     grad_u: torch.Tensor,
    52:     grad_v: torch.Tensor,
    53:     state: dict[str, Any],
    54:     hyperparameters: dict[str, Any],
    55: ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    56:     """Adam update step WITHOUT bias correction."""
    57:     lr = float(hyperparameters["lr"])
    58:     beta1 = float(hyperparameters["beta1"])
    59:     beta2 = float(hyperparameters["beta2"])
    60:     eps = float(hyperparameters["eps"])
    61:     t = state["t"] + 1
    62:     m_u = beta1 * state["m_u"] + (1.0 - beta1) * grad_u
    63:     s_u = beta2 * state["s_u"] + (1.0 - beta2) * grad_u * grad_u
    64:     u_new = u - lr * m_u / (torch.sqrt(s_u) + eps)
    65:     m_v = beta1 * state["m_v"] + (1.0 - beta1) * grad_v
    66:     s_v = beta2 * state["s_v"] + (1.0 - beta2) * grad_v * grad_v
    67:     v_new = v - lr * m_v / (torch.sqrt(s_v) + eps)
    68:     return u_new, v_new, {"t": t, "m_u": m_u, "s_u": s_u, "m_v": m_v, "s_v": s_v}
    69: if __name__ == "__main__":
    70:     run_cli(
    71:         get_hyperparameters=get_hyperparameters,
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
