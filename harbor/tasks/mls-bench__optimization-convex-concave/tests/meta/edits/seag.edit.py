"""Official SEAG baseline for optimization-convex-concave."""

_FILE = "RAIN/optimization_convex_concave/custom_strategy.py"

_CONTENT = '''\
def init_state(
    problem: ProblemSpec,
    initial_z: np.ndarray,
    seed: int,
    hyperparameters: dict[str, Any],
) -> dict[str, Any]:
    z0 = as_vector(initial_z, expected_dim=2 * problem.dim)
    return {
        "z": z0,
        "anchor_z": z0.copy(),
        "step_index": 0,
    }


def step(
    state: dict[str, Any],
    oracle: StochasticOracle,
    problem: ProblemSpec,
    hyperparameters: dict[str, Any],
    max_sfo_calls: int,
) -> StepOutput:
    tau = float(hyperparameters["tau"])
    z = as_vector(state["z"], expected_dim=2 * problem.dim)
    anchor_z = as_vector(state["anchor_z"], expected_dim=2 * problem.dim)
    step_index = int(state.get("step_index", 0))
    coeff = 1.0 / (step_index + 3.0)

    g = oracle.grad(z)
    w = z - tau * g + coeff * (anchor_z - z) + oracle.noise()
    gw = oracle.grad(w)
    z_next = z - tau * gw + coeff * (anchor_z - z) + oracle.noise()
    metric_iterate = z_next if problem.name == "bilinear" else z
    return make_step_output(
        {"z": z_next, "anchor_z": anchor_z, "step_index": step_index + 1},
        metric_iterate,
        2,
    )


def get_hyperparameters(problem_name: str, sigma: float) -> dict[str, Any]:
    if problem_name == "bilinear":
        return {"tau": 0.1}
    if problem_name == "delta_nu":
        return {"tau": 1.0}
    raise KeyError(f"Unknown problem: {problem_name}")
'''

OPS = [{"op": "replace", "file": _FILE, "start_line": 24, "end_line": 75, "content": _CONTENT}]
