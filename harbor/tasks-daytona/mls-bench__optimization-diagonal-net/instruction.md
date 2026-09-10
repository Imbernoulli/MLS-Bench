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

1. `get_hyperparameters(dim, sparsity, delta, alpha_init)` — return optimizer configuration.
2. `init_state(u, v, hyperparameters)` — initialise optimizer state.
3. `step(u, v, grad_u, grad_v, state, hyperparameters)` — perform one update step.

The default template implements vanilla gradient descent. Your goal is to reliably recover the sparse ground truth (low test MSE) from fewer training samples.

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


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/RAIN/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits that change code outside these ranges — or creating new files, or
deleting whole files — will cause your submission to be invalid.

The line numbers mark an editable **region**, not a fixed line-count budget: you
may add or remove lines inside it. Only code outside the editable ranges must
stay unchanged.

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
    27:     alpha_init: float,
    28: ) -> dict[str, Any]:
    29:     """Return optimizer hyperparameters for this problem setting.
    30: 
    31:     Args:
    32:         dim: ambient dimension d.
    33:         sparsity: number of nonzero entries k in the ground truth.
    34:         delta: Rademacher noise magnitude (±delta) added to labels each step.
    35:         alpha_init: initialisation scale α of the diagonal net; u and v both
    36:             start at α / sqrt(2d).  Small α puts training in the rich regime
    37:             (sparsity-inducing implicit bias), large α in the lazy / kernel
    38:             regime (ℓ2-like bias) — the best optimizer may differ between them.
    39: 
    40:     Returns:
    41:         dict of hyperparameters used by init_state and step.
    42:     """
    43:     return {"lr": 0.01}
    44: 
    45: 
    46: def init_state(
    47:     u: torch.Tensor,
    48:     v: torch.Tensor,
    49:     hyperparameters: dict[str, Any],
    50: ) -> dict[str, Any]:
    51:     """Initialise optimizer state from the model parameters u, v.
    52: 
    53:     Args:
    54:         u: initial parameter vector u (shape (d,), float64).
    55:         v: initial parameter vector v (shape (d,), float64).
    56:         hyperparameters: dict from get_hyperparameters.
    57: 
    58:     Returns:
    59:         dict of optimizer state (passed to step and updated each iteration).
    60:     """
    61:     return {"t": 0}
    62: 
    63: 
    64: def step(
    65:     u: torch.Tensor,
    66:     v: torch.Tensor,
    67:     grad_u: torch.Tensor,
    68:     grad_v: torch.Tensor,
    69:     state: dict[str, Any],
    70:     hyperparameters: dict[str, Any],
    71: ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    72:     """Perform one optimizer step.
    73: 
    74:     Args:
    75:         u: current parameter u (shape (d,), float64).
    76:         v: current parameter v (shape (d,), float64).
    77:         grad_u: gradient of MSE loss w.r.t. u (shape (d,), float64).
    78:         grad_v: gradient of MSE loss w.r.t. v (shape (d,), float64).
    79:         state: mutable optimizer state from init_state / previous step.
    80:         hyperparameters: dict from get_hyperparameters.
    81: 
    82:     Returns:
    83:         (u_new, v_new, state_new) tuple of updated parameters and state.
    84:     """
    85:     lr = float(hyperparameters["lr"])
    86:     state["t"] = state.get("t", 0) + 1
    87:     return u - lr * grad_u, v - lr * grad_v, state
    88: 
    89: 
    90: 
    91: def main() -> None:
    92:     """FIXED entry: invoked by the wrapper scripts/fixed_entry.py (which
    93:     preloads and unlinks the staged input blobs before this module is
    94:     imported) or via a direct launch of this file."""
    95:     run_cli(
    96:         get_hyperparameters=get_hyperparameters,
    97:         init_state=init_state,
    98:         step=step,
    99:     )
   100: 
   101: 
   102: if __name__ == "__main__":
   103:     main()
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `sgd` baseline — editable region  [READ-ONLY — reference implementation]

In `RAIN/opt_diagonal_net/custom_optimizer.py`:

```python
Lines 23–53:
    20: # =====================================================================
    21: 
    22: 
    23: def get_hyperparameters(
    24:     dim: int,
    25:     sparsity: int,
    26:     delta: float,
    27:     alpha_init: float,
    28: ) -> dict[str, Any]:
    29:     """SGD hyperparameters: lr=0.1."""
    30:     return {"lr": 0.1}
    31: 
    32: 
    33: def init_state(
    34:     u: torch.Tensor,
    35:     v: torch.Tensor,
    36:     hyperparameters: dict[str, Any],
    37: ) -> dict[str, Any]:
    38:     """SGD requires no additional state."""
    39:     return {"t": 0}
    40: 
    41: 
    42: def step(
    43:     u: torch.Tensor,
    44:     v: torch.Tensor,
    45:     grad_u: torch.Tensor,
    46:     grad_v: torch.Tensor,
    47:     state: dict[str, Any],
    48:     hyperparameters: dict[str, Any],
    49: ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    50:     """Vanilla gradient descent step."""
    51:     lr = float(hyperparameters["lr"])
    52:     state["t"] = state.get("t", 0) + 1
    53:     return u - lr * grad_u, v - lr * grad_v, state
    54: def main() -> None:
    55:     """FIXED entry: invoked by the wrapper scripts/fixed_entry.py (which
    56:     preloads and unlinks the staged input blobs before this module is
```

### `adagrad` baseline — editable region  [READ-ONLY — reference implementation]

In `RAIN/opt_diagonal_net/custom_optimizer.py`:

```python
Lines 23–62:
    20: # =====================================================================
    21: 
    22: 
    23: def get_hyperparameters(
    24:     dim: int,
    25:     sparsity: int,
    26:     delta: float,
    27:     alpha_init: float,
    28: ) -> dict[str, Any]:
    29:     """AdaGrad hyperparameters: lr=0.01, eps=1e-6."""
    30:     return {"lr": 0.01, "eps": 1e-6}
    31: 
    32: 
    33: def init_state(
    34:     u: torch.Tensor,
    35:     v: torch.Tensor,
    36:     hyperparameters: dict[str, Any],
    37: ) -> dict[str, Any]:
    38:     """AdaGrad state: accumulated squared gradients."""
    39:     d = u.shape[0]
    40:     return {
    41:         "t": 0,
    42:         "g_sum_u": torch.zeros(d, dtype=torch.float64),
    43:         "g_sum_v": torch.zeros(d, dtype=torch.float64),
    44:     }
    45: 
    46: 
    47: def step(
    48:     u: torch.Tensor,
    49:     v: torch.Tensor,
    50:     grad_u: torch.Tensor,
    51:     grad_v: torch.Tensor,
    52:     state: dict[str, Any],
    53:     hyperparameters: dict[str, Any],
    54: ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    55:     """AdaGrad update step."""
    56:     lr = float(hyperparameters["lr"])
    57:     eps = float(hyperparameters["eps"])
    58:     g_sum_u = state["g_sum_u"] + grad_u * grad_u
    59:     g_sum_v = state["g_sum_v"] + grad_v * grad_v
    60:     u_new = u - lr * grad_u / (torch.sqrt(g_sum_u) + eps)
    61:     v_new = v - lr * grad_v / (torch.sqrt(g_sum_v) + eps)
    62:     return u_new, v_new, {"t": state["t"] + 1, "g_sum_u": g_sum_u, "g_sum_v": g_sum_v}
    63: def main() -> None:
    64:     """FIXED entry: invoked by the wrapper scripts/fixed_entry.py (which
    65:     preloads and unlinks the staged input blobs before this module is
```

### `adam` baseline — editable region  [READ-ONLY — reference implementation]

In `RAIN/opt_diagonal_net/custom_optimizer.py`:

```python
Lines 23–69:
    20: # =====================================================================
    21: 
    22: 
    23: def get_hyperparameters(
    24:     dim: int,
    25:     sparsity: int,
    26:     delta: float,
    27:     alpha_init: float,
    28: ) -> dict[str, Any]:
    29:     """Adam (no bias correction) hyperparameters: lr=0.05, beta2=0.999."""
    30:     return {"lr": 0.05, "beta1": 0.9, "beta2": 0.999, "eps": 1e-6}
    31: 
    32: 
    33: def init_state(
    34:     u: torch.Tensor,
    35:     v: torch.Tensor,
    36:     hyperparameters: dict[str, Any],
    37: ) -> dict[str, Any]:
    38:     """Adam state: first and second moment estimates."""
    39:     d = u.shape[0]
    40:     return {
    41:         "t": 0,
    42:         "m_u": torch.zeros(d, dtype=torch.float64),
    43:         "s_u": torch.zeros(d, dtype=torch.float64),
    44:         "m_v": torch.zeros(d, dtype=torch.float64),
    45:         "s_v": torch.zeros(d, dtype=torch.float64),
    46:     }
    47: 
    48: 
    49: def step(
    50:     u: torch.Tensor,
    51:     v: torch.Tensor,
    52:     grad_u: torch.Tensor,
    53:     grad_v: torch.Tensor,
    54:     state: dict[str, Any],
    55:     hyperparameters: dict[str, Any],
    56: ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    57:     """Adam update step WITHOUT bias correction."""
    58:     lr = float(hyperparameters["lr"])
    59:     beta1 = float(hyperparameters["beta1"])
    60:     beta2 = float(hyperparameters["beta2"])
    61:     eps = float(hyperparameters["eps"])
    62:     t = state["t"] + 1
    63:     m_u = beta1 * state["m_u"] + (1.0 - beta1) * grad_u
    64:     s_u = beta2 * state["s_u"] + (1.0 - beta2) * grad_u * grad_u
    65:     u_new = u - lr * m_u / (torch.sqrt(s_u) + eps)
    66:     m_v = beta1 * state["m_v"] + (1.0 - beta1) * grad_v
    67:     s_v = beta2 * state["s_v"] + (1.0 - beta2) * grad_v * grad_v
    68:     v_new = v - lr * m_v / (torch.sqrt(s_v) + eps)
    69:     return u_new, v_new, {"t": t, "m_u": m_u, "s_u": s_u, "m_v": m_v, "s_v": s_v}
    70: def main() -> None:
    71:     """FIXED entry: invoked by the wrapper scripts/fixed_entry.py (which
    72:     preloads and unlinks the staged input blobs before this module is
```

### `adam2` baseline — editable region  [READ-ONLY — reference implementation]

In `RAIN/opt_diagonal_net/custom_optimizer.py`:

```python
Lines 23–69:
    20: # =====================================================================
    21: 
    22: 
    23: def get_hyperparameters(
    24:     dim: int,
    25:     sparsity: int,
    26:     delta: float,
    27:     alpha_init: float,
    28: ) -> dict[str, Any]:
    29:     """Adam (no bias correction) hyperparameters: lr=0.1, beta2=0.95."""
    30:     return {"lr": 0.1, "beta1": 0.9, "beta2": 0.95, "eps": 1e-6}
    31: 
    32: 
    33: def init_state(
    34:     u: torch.Tensor,
    35:     v: torch.Tensor,
    36:     hyperparameters: dict[str, Any],
    37: ) -> dict[str, Any]:
    38:     """Adam state: first and second moment estimates."""
    39:     d = u.shape[0]
    40:     return {
    41:         "t": 0,
    42:         "m_u": torch.zeros(d, dtype=torch.float64),
    43:         "s_u": torch.zeros(d, dtype=torch.float64),
    44:         "m_v": torch.zeros(d, dtype=torch.float64),
    45:         "s_v": torch.zeros(d, dtype=torch.float64),
    46:     }
    47: 
    48: 
    49: def step(
    50:     u: torch.Tensor,
    51:     v: torch.Tensor,
    52:     grad_u: torch.Tensor,
    53:     grad_v: torch.Tensor,
    54:     state: dict[str, Any],
    55:     hyperparameters: dict[str, Any],
    56: ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    57:     """Adam update step WITHOUT bias correction."""
    58:     lr = float(hyperparameters["lr"])
    59:     beta1 = float(hyperparameters["beta1"])
    60:     beta2 = float(hyperparameters["beta2"])
    61:     eps = float(hyperparameters["eps"])
    62:     t = state["t"] + 1
    63:     m_u = beta1 * state["m_u"] + (1.0 - beta1) * grad_u
    64:     s_u = beta2 * state["s_u"] + (1.0 - beta2) * grad_u * grad_u
    65:     u_new = u - lr * m_u / (torch.sqrt(s_u) + eps)
    66:     m_v = beta1 * state["m_v"] + (1.0 - beta1) * grad_v
    67:     s_v = beta2 * state["s_v"] + (1.0 - beta2) * grad_v * grad_v
    68:     v_new = v - lr * m_v / (torch.sqrt(s_v) + eps)
    69:     return u_new, v_new, {"t": t, "m_u": m_u, "s_u": s_u, "m_v": m_v, "s_v": s_v}
    70: def main() -> None:
    71:     """FIXED entry: invoked by the wrapper scripts/fixed_entry.py (which
    72:     preloads and unlinks the staged input blobs before this module is
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

## Time Budget

You have **5 hours** of wall-clock time before submission, covering
everything you do here: reading the code, editing it, and any trial runs
you launch.

Good luck.
