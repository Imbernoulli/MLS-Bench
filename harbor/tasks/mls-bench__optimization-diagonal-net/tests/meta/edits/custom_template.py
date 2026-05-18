"""Editable optimizer scaffold for the opt-diagonal-net MLS-Bench task.

Implement a custom optimizer for training a diagonal-net model to recover
a sparse linear predictor.  You may edit the three functions below
(get_hyperparameters, init_state, step) while the benchmark harness,
data generation, model, stopping rule, and search protocol are fixed.
"""

from __future__ import annotations

from typing import Any

import torch

from fixed_benchmark import run_cli


# =====================================================================
# EDITABLE: get_hyperparameters, init_state, step  (lines 23 to 90)
# =====================================================================


def get_hyperparameters(
    dim: int,
    sparsity: int,
    delta: float,
) -> dict[str, Any]:
    """Return optimizer hyperparameters for this problem setting.

    Args:
        dim: ambient dimension d.
        sparsity: number of nonzero entries k in the ground truth.
        delta: Rademacher noise magnitude (±delta) added to labels each step.

    Returns:
        dict of hyperparameters used by init_state and step.
    """
    return {"lr": 0.01}


def init_state(
    u: torch.Tensor,
    v: torch.Tensor,
    hyperparameters: dict[str, Any],
) -> dict[str, Any]:
    """Initialise optimizer state from the model parameters u, v.

    Args:
        u: initial parameter vector u (shape (d,), float64).
        v: initial parameter vector v (shape (d,), float64).
        hyperparameters: dict from get_hyperparameters.

    Returns:
        dict of optimizer state (passed to step and updated each iteration).
    """
    return {"t": 0}


def step(
    u: torch.Tensor,
    v: torch.Tensor,
    grad_u: torch.Tensor,
    grad_v: torch.Tensor,
    state: dict[str, Any],
    hyperparameters: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Perform one optimizer step.

    Args:
        u: current parameter u (shape (d,), float64).
        v: current parameter v (shape (d,), float64).
        grad_u: gradient of MSE loss w.r.t. u (shape (d,), float64).
        grad_v: gradient of MSE loss w.r.t. v (shape (d,), float64).
        state: mutable optimizer state from init_state / previous step.
        hyperparameters: dict from get_hyperparameters.

    Returns:
        (u_new, v_new, state_new) tuple of updated parameters and state.
    """
    lr = float(hyperparameters["lr"])
    state["t"] = state.get("t", 0) + 1
    return u - lr * grad_u, v - lr * grad_v, state








if __name__ == "__main__":
    run_cli(
        get_hyperparameters=get_hyperparameters,
        init_state=init_state,
        step=step,
    )
