"""Editable strategy scaffold for the optimization-convex-concave MLS-Bench task."""

from __future__ import annotations

from typing import Any

import numpy as np

from fixed_benchmark import (
    ProblemSpec,
    StepOutput,
    StochasticOracle,
    as_vector,
    make_step_output,
    run_cli,
)


# =====================================================================
# EDITABLE: init_state, step, get_hyperparameters
# =====================================================================


def init_state(
    problem: ProblemSpec,
    initial_z: np.ndarray,
    seed: int,
    hyperparameters: dict[str, Any],
) -> dict[str, Any]:
    """Initialize algorithm state from the fixed starting point."""
    return {
        "z": as_vector(initial_z, expected_dim=2 * problem.dim),
        "step_index": 0,
    }


def step(
    state: dict[str, Any],
    oracle: StochasticOracle,
    problem: ProblemSpec,
    hyperparameters: dict[str, Any],
    max_sfo_calls: int,
) -> StepOutput:
    """Default baseline: the official SEG / EG update from the MATLAB scripts."""
    tau = float(hyperparameters["tau"])
    z = as_vector(state["z"], expected_dim=2 * problem.dim)
    step_index = int(state.get("step_index", 0))

    g = oracle.grad(z)
    w = z - tau * g + oracle.noise()
    gw = oracle.grad(w)
    z_next = z - tau * gw + oracle.noise()
    metric_iterate = z_next if problem.name == "bilinear" else z
    return make_step_output(
        {"z": z_next, "step_index": step_index + 1},
        metric_iterate,
        2,
    )


def get_hyperparameters(problem_name: str, sigma: float) -> dict[str, Any]:
    """Return the official per-problem step size."""
    if problem_name == "bilinear":
        return {"tau": 0.1}
    if problem_name == "delta_nu":
        return {"tau": 1.0}
    raise KeyError(f"Unknown problem: {problem_name}")








if __name__ == "__main__":
    run_cli(
        init_state=init_state,
        step=step,
        get_hyperparameters=get_hyperparameters,
    )
