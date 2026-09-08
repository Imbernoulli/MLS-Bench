"""Official RAIN baseline for optimization-convex-concave."""

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
        "step_index": 0,
        "weight_sum": 0.0,
        "weighted_flow_sum": np.zeros_like(z0),
    }


def step(
    state: dict[str, Any],
    oracle: StochasticOracle,
    problem: ProblemSpec,
    hyperparameters: dict[str, Any],
    max_sfo_calls: int,
) -> StepOutput:
    tau = float(hyperparameters["tau"])
    lam = float(hyperparameters["lambda"])
    gamma = float(hyperparameters["gamma"])
    z = as_vector(state["z"], expected_dim=2 * problem.dim)
    step_index = int(state.get("step_index", 0))
    weight_sum = float(state.get("weight_sum", 0.0))
    weighted_flow_sum = as_vector(state.get("weighted_flow_sum", np.zeros_like(z)), expected_dim=2 * problem.dim)

    g = oracle.grad(z)
    anchor_z = tau * lam * (weighted_flow_sum - weight_sum * z)
    w = z - tau * g + anchor_z + oracle.noise()
    gw = oracle.grad(w)
    anchor_w = tau * lam * (weighted_flow_sum - weight_sum * w)
    z_next = z - tau * gw + anchor_w + oracle.noise()

    current_weight = gamma * (1.0 + gamma) ** (step_index + 1)
    next_state = {
        "z": z_next,
        "step_index": step_index + 1,
        "weight_sum": weight_sum + current_weight,
        "weighted_flow_sum": weighted_flow_sum + current_weight * z_next,
    }
    metric_iterate = z_next if problem.name == "bilinear" else z
    return make_step_output(next_state, metric_iterate, 2)


def get_hyperparameters(problem_name: str, sigma: float) -> dict[str, Any]:
    if problem_name == "bilinear":
        return {"tau": 0.1, "lambda": 0.1, "gamma": 0.001}
    if problem_name == "delta_nu":
        return {"tau": 1.0, "lambda": 0.01, "gamma": 0.0001}
    raise KeyError(f"Unknown problem: {problem_name}")
'''

OPS = [{"op": "replace", "file": _FILE, "start_line": 24, "end_line": 75, "content": _CONTENT}]
