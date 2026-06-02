# MLS-Bench: optimization-convex-concave

# RAIN Convex-Concave

## Research Question
Can you improve convergence on the convex-concave saddle-point benchmark instances used by the official RAIN repository?

## Background
Convex-concave saddle-point problems `min_x max_y F(x, y)` are a canonical model for minimax optimization (game-theoretic equilibria, robust learning, GANs). Even simple bilinear instances `f(x, y) = xy` make naive simultaneous gradient descent-ascent diverge; extragradient (Korpelevich, 1976), optimistic methods, and noise-robust analogues are needed. The RAIN reference codebase exercises two regimes — a scalar bilinear problem and a structured `(delta, nu)`-strongly-monotone problem — both subject to additive Gaussian update noise.

## What You Can Modify
Edit only the scaffold file `RAIN/optimization_convex_concave/custom_strategy.py` inside the editable block containing:

1. `init_state(problem, initial_z, seed, hyperparameters)`
2. `step(state, oracle, problem, hyperparameters, max_sfo_calls)`
3. `get_hyperparameters(problem_name, sigma)`

The benchmark harness, problem definitions, update-noise model, official iteration counts, initializations, and metric computation are fixed.

## Interface Notes
- `init_state(...)` must preserve the provided starting point in `state["z"]`.
- `step(...)` should implement one official-style iteration of the chosen method.
- The oracle exposes deterministic gradients and fixed-scale Gaussian update noise so the update equations can match the MATLAB scripts directly.
- `get_hyperparameters(...)` should return the per-problem constants used by the method.

## Baselines (reference implementations from the RAIN repo)
- **SEG** — Stochastic Extragradient (Korpelevich, 1976; modern stochastic analyses include Mishchenko et al., AISTATS 2020).
- **R-SEG** — restarted SEG variant from the RAIN repo.
- **SEAG** — Stochastic Extra-Anchored Gradient (RAIN-companion method).
- **RAIN** — the repo's main proposed update (anchor-iteration noise-robust method).

## Read-Only References
- `RAIN/README.md`
- `RAIN/src/bilinear_func/exp_gnorm.m`
- `RAIN/src/delta_func/exp_gnorm.m`

These are the primary references; the task follows them directly rather than the earlier MLS-Bench-specific generalized variant.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/RAIN/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `RAIN/optimization_convex_concave/custom_strategy.py`
- editable lines **24–75**




## Readable Context


### `RAIN/optimization_convex_concave/custom_strategy.py`  [EDITABLE — lines 24–75 only]

```python
     1: """Editable strategy scaffold for the optimization-convex-concave MLS-Bench task."""
     2: 
     3: from __future__ import annotations
     4: 
     5: from typing import Any
     6: 
     7: import numpy as np
     8: 
     9: from fixed_benchmark import (
    10:     ProblemSpec,
    11:     StepOutput,
    12:     StochasticOracle,
    13:     as_vector,
    14:     make_step_output,
    15:     run_cli,
    16: )
    17: 
    18: 
    19: # =====================================================================
    20: # EDITABLE: init_state, step, get_hyperparameters
    21: # =====================================================================
    22: 
    23: 
    24: def init_state(
    25:     problem: ProblemSpec,
    26:     initial_z: np.ndarray,
    27:     seed: int,
    28:     hyperparameters: dict[str, Any],
    29: ) -> dict[str, Any]:
    30:     """Initialize algorithm state from the fixed starting point."""
    31:     return {
    32:         "z": as_vector(initial_z, expected_dim=2 * problem.dim),
    33:         "step_index": 0,
    34:     }
    35: 
    36: 
    37: def step(
    38:     state: dict[str, Any],
    39:     oracle: StochasticOracle,
    40:     problem: ProblemSpec,
    41:     hyperparameters: dict[str, Any],
    42:     max_sfo_calls: int,
    43: ) -> StepOutput:
    44:     """Default baseline: the official SEG / EG update from the MATLAB scripts."""
    45:     tau = float(hyperparameters["tau"])
    46:     z = as_vector(state["z"], expected_dim=2 * problem.dim)
    47:     step_index = int(state.get("step_index", 0))
    48: 
    49:     g = oracle.grad(z)
    50:     w = z - tau * g + oracle.noise()
    51:     gw = oracle.grad(w)
    52:     z_next = z - tau * gw + oracle.noise()
    53:     metric_iterate = z_next if problem.name == "bilinear" else z
    54:     return make_step_output(
    55:         {"z": z_next, "step_index": step_index + 1},
    56:         metric_iterate,
    57:         2,
    58:     )
    59: 
    60: 
    61: def get_hyperparameters(problem_name: str, sigma: float) -> dict[str, Any]:
    62:     """Return the official per-problem step size."""
    63:     if problem_name == "bilinear":
    64:         return {"tau": 0.1}
    65:     if problem_name == "delta_nu":
    66:         return {"tau": 1.0}
    67:     raise KeyError(f"Unknown problem: {problem_name}")
    68: 
    69: 
    70: 
    71: 
    72: 
    73: 
    74: 
    75: 
    76: if __name__ == "__main__":
    77:     run_cli(
    78:         init_state=init_state,
    79:         step=step,
    80:         get_hyperparameters=get_hyperparameters,
    81:     )
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `seg` baseline — editable region  [READ-ONLY — reference implementation]

In `RAIN/optimization_convex_concave/custom_strategy.py`:

```python
Lines 24–64:
    21: # =====================================================================
    22: 
    23: 
    24: def init_state(
    25:     problem: ProblemSpec,
    26:     initial_z: np.ndarray,
    27:     seed: int,
    28:     hyperparameters: dict[str, Any],
    29: ) -> dict[str, Any]:
    30:     return {
    31:         "z": as_vector(initial_z, expected_dim=2 * problem.dim),
    32:         "step_index": 0,
    33:     }
    34: 
    35: 
    36: def step(
    37:     state: dict[str, Any],
    38:     oracle: StochasticOracle,
    39:     problem: ProblemSpec,
    40:     hyperparameters: dict[str, Any],
    41:     max_sfo_calls: int,
    42: ) -> StepOutput:
    43:     tau = float(hyperparameters["tau"])
    44:     z = as_vector(state["z"], expected_dim=2 * problem.dim)
    45:     step_index = int(state.get("step_index", 0))
    46: 
    47:     g = oracle.grad(z)
    48:     w = z - tau * g + oracle.noise()
    49:     gw = oracle.grad(w)
    50:     z_next = z - tau * gw + oracle.noise()
    51:     metric_iterate = z_next if problem.name == "bilinear" else z
    52:     return make_step_output(
    53:         {"z": z_next, "step_index": step_index + 1},
    54:         metric_iterate,
    55:         2,
    56:     )
    57: 
    58: 
    59: def get_hyperparameters(problem_name: str, sigma: float) -> dict[str, Any]:
    60:     if problem_name == "bilinear":
    61:         return {"tau": 0.1}
    62:     if problem_name == "delta_nu":
    63:         return {"tau": 1.0}
    64:     raise KeyError(f"Unknown problem: {problem_name}")
    65: if __name__ == "__main__":
    66:     run_cli(
    67:         init_state=init_state,
```

### `r_seg` baseline — editable region  [READ-ONLY — reference implementation]

In `RAIN/optimization_convex_concave/custom_strategy.py`:

```python
Lines 24–68:
    21: # =====================================================================
    22: 
    23: 
    24: def init_state(
    25:     problem: ProblemSpec,
    26:     initial_z: np.ndarray,
    27:     seed: int,
    28:     hyperparameters: dict[str, Any],
    29: ) -> dict[str, Any]:
    30:     z0 = as_vector(initial_z, expected_dim=2 * problem.dim)
    31:     return {
    32:         "z": z0,
    33:         "anchor_z": z0.copy(),
    34:         "step_index": 0,
    35:     }
    36: 
    37: 
    38: def step(
    39:     state: dict[str, Any],
    40:     oracle: StochasticOracle,
    41:     problem: ProblemSpec,
    42:     hyperparameters: dict[str, Any],
    43:     max_sfo_calls: int,
    44: ) -> StepOutput:
    45:     tau = float(hyperparameters["tau"])
    46:     lam = float(hyperparameters["lambda"])
    47:     z = as_vector(state["z"], expected_dim=2 * problem.dim)
    48:     anchor_z = as_vector(state["anchor_z"], expected_dim=2 * problem.dim)
    49:     step_index = int(state.get("step_index", 0))
    50: 
    51:     g = oracle.grad(z)
    52:     w = z - tau * g + tau * lam * (anchor_z - z) + oracle.noise()
    53:     gw = oracle.grad(w)
    54:     z_next = z - tau * gw + tau * lam * (anchor_z - w) + oracle.noise()
    55:     metric_iterate = z_next if problem.name == "bilinear" else z
    56:     return make_step_output(
    57:         {"z": z_next, "anchor_z": anchor_z, "step_index": step_index + 1},
    58:         metric_iterate,
    59:         2,
    60:     )
    61: 
    62: 
    63: def get_hyperparameters(problem_name: str, sigma: float) -> dict[str, Any]:
    64:     if problem_name == "bilinear":
    65:         return {"tau": 0.1, "lambda": 0.1}
    66:     if problem_name == "delta_nu":
    67:         return {"tau": 1.0, "lambda": 0.01}
    68:     raise KeyError(f"Unknown problem: {problem_name}")
    69: if __name__ == "__main__":
    70:     run_cli(
    71:         init_state=init_state,
```

### `seag` baseline — editable region  [READ-ONLY — reference implementation]

In `RAIN/optimization_convex_concave/custom_strategy.py`:

```python
Lines 24–68:
    21: # =====================================================================
    22: 
    23: 
    24: def init_state(
    25:     problem: ProblemSpec,
    26:     initial_z: np.ndarray,
    27:     seed: int,
    28:     hyperparameters: dict[str, Any],
    29: ) -> dict[str, Any]:
    30:     z0 = as_vector(initial_z, expected_dim=2 * problem.dim)
    31:     return {
    32:         "z": z0,
    33:         "anchor_z": z0.copy(),
    34:         "step_index": 0,
    35:     }
    36: 
    37: 
    38: def step(
    39:     state: dict[str, Any],
    40:     oracle: StochasticOracle,
    41:     problem: ProblemSpec,
    42:     hyperparameters: dict[str, Any],
    43:     max_sfo_calls: int,
    44: ) -> StepOutput:
    45:     tau = float(hyperparameters["tau"])
    46:     z = as_vector(state["z"], expected_dim=2 * problem.dim)
    47:     anchor_z = as_vector(state["anchor_z"], expected_dim=2 * problem.dim)
    48:     step_index = int(state.get("step_index", 0))
    49:     coeff = 1.0 / (step_index + 3.0)
    50: 
    51:     g = oracle.grad(z)
    52:     w = z - tau * g + coeff * (anchor_z - z) + oracle.noise()
    53:     gw = oracle.grad(w)
    54:     z_next = z - tau * gw + coeff * (anchor_z - z) + oracle.noise()
    55:     metric_iterate = z_next if problem.name == "bilinear" else z
    56:     return make_step_output(
    57:         {"z": z_next, "anchor_z": anchor_z, "step_index": step_index + 1},
    58:         metric_iterate,
    59:         2,
    60:     )
    61: 
    62: 
    63: def get_hyperparameters(problem_name: str, sigma: float) -> dict[str, Any]:
    64:     if problem_name == "bilinear":
    65:         return {"tau": 0.1}
    66:     if problem_name == "delta_nu":
    67:         return {"tau": 1.0}
    68:     raise KeyError(f"Unknown problem: {problem_name}")
    69: if __name__ == "__main__":
    70:     run_cli(
    71:         init_state=init_state,
```

### `rain` baseline — editable region  [READ-ONLY — reference implementation]

In `RAIN/optimization_convex_concave/custom_strategy.py`:

```python
Lines 24–77:
    21: # =====================================================================
    22: 
    23: 
    24: def init_state(
    25:     problem: ProblemSpec,
    26:     initial_z: np.ndarray,
    27:     seed: int,
    28:     hyperparameters: dict[str, Any],
    29: ) -> dict[str, Any]:
    30:     z0 = as_vector(initial_z, expected_dim=2 * problem.dim)
    31:     return {
    32:         "z": z0,
    33:         "step_index": 0,
    34:         "weight_sum": 0.0,
    35:         "weighted_flow_sum": np.zeros_like(z0),
    36:     }
    37: 
    38: 
    39: def step(
    40:     state: dict[str, Any],
    41:     oracle: StochasticOracle,
    42:     problem: ProblemSpec,
    43:     hyperparameters: dict[str, Any],
    44:     max_sfo_calls: int,
    45: ) -> StepOutput:
    46:     tau = float(hyperparameters["tau"])
    47:     lam = float(hyperparameters["lambda"])
    48:     gamma = float(hyperparameters["gamma"])
    49:     z = as_vector(state["z"], expected_dim=2 * problem.dim)
    50:     step_index = int(state.get("step_index", 0))
    51:     weight_sum = float(state.get("weight_sum", 0.0))
    52:     weighted_flow_sum = as_vector(state.get("weighted_flow_sum", np.zeros_like(z)), expected_dim=2 * problem.dim)
    53: 
    54:     g = oracle.grad(z)
    55:     anchor_z = tau * lam * (weighted_flow_sum - weight_sum * z)
    56:     w = z - tau * g + anchor_z + oracle.noise()
    57:     gw = oracle.grad(w)
    58:     anchor_w = tau * lam * (weighted_flow_sum - weight_sum * w)
    59:     z_next = z - tau * gw + anchor_w + oracle.noise()
    60: 
    61:     current_weight = gamma * (1.0 + gamma) ** (step_index + 1)
    62:     next_state = {
    63:         "z": z_next,
    64:         "step_index": step_index + 1,
    65:         "weight_sum": weight_sum + current_weight,
    66:         "weighted_flow_sum": weighted_flow_sum + current_weight * z_next,
    67:     }
    68:     metric_iterate = z_next if problem.name == "bilinear" else z
    69:     return make_step_output(next_state, metric_iterate, 2)
    70: 
    71: 
    72: def get_hyperparameters(problem_name: str, sigma: float) -> dict[str, Any]:
    73:     if problem_name == "bilinear":
    74:         return {"tau": 0.1, "lambda": 0.1, "gamma": 0.001}
    75:     if problem_name == "delta_nu":
    76:         return {"tau": 1.0, "lambda": 0.01, "gamma": 0.0001}
    77:     raise KeyError(f"Unknown problem: {problem_name}")
    78: if __name__ == "__main__":
    79:     run_cli(
    80:         init_state=init_state,
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
