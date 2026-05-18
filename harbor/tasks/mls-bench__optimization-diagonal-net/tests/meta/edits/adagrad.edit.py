"""AdaGrad baseline (lr=0.01, eps=1e-6) for opt-diagonal-net."""

_FILE = "RAIN/opt_diagonal_net/custom_optimizer.py"

_CONTENT = '''\
def get_hyperparameters(
    dim: int,
    sparsity: int,
    delta: float,
) -> dict[str, Any]:
    """AdaGrad hyperparameters: lr=0.01, eps=1e-6."""
    return {"lr": 0.01, "eps": 1e-6}


def init_state(
    u: torch.Tensor,
    v: torch.Tensor,
    hyperparameters: dict[str, Any],
) -> dict[str, Any]:
    """AdaGrad state: accumulated squared gradients."""
    d = u.shape[0]
    return {
        "t": 0,
        "g_sum_u": torch.zeros(d, dtype=torch.float64),
        "g_sum_v": torch.zeros(d, dtype=torch.float64),
    }


def step(
    u: torch.Tensor,
    v: torch.Tensor,
    grad_u: torch.Tensor,
    grad_v: torch.Tensor,
    state: dict[str, Any],
    hyperparameters: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """AdaGrad update step."""
    lr = float(hyperparameters["lr"])
    eps = float(hyperparameters["eps"])
    g_sum_u = state["g_sum_u"] + grad_u * grad_u
    g_sum_v = state["g_sum_v"] + grad_v * grad_v
    u_new = u - lr * grad_u / (torch.sqrt(g_sum_u) + eps)
    v_new = v - lr * grad_v / (torch.sqrt(g_sum_v) + eps)
    return u_new, v_new, {"t": state["t"] + 1, "g_sum_u": g_sum_u, "g_sum_v": g_sum_v}
'''

OPS = [{"op": "replace", "file": _FILE, "start_line": 23, "end_line": 90, "content": _CONTENT}]
