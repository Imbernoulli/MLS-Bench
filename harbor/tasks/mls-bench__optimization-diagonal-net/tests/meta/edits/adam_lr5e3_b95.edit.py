"""Adam (no bias correction) baseline (lr=0.005, beta2=0.95, eps=1e-6) for opt-diagonal-net."""

_FILE = "RAIN/opt_diagonal_net/custom_optimizer.py"

_CONTENT = '''\
def get_hyperparameters(
    dim: int,
    sparsity: int,
    delta: float,
) -> dict[str, Any]:
    """Adam (no bias correction) hyperparameters: lr=0.005, beta2=0.95."""
    return {"lr": 0.005, "beta1": 0.9, "beta2": 0.95, "eps": 1e-6}


def init_state(
    u: torch.Tensor,
    v: torch.Tensor,
    hyperparameters: dict[str, Any],
) -> dict[str, Any]:
    """Adam state: first and second moment estimates."""
    d = u.shape[0]
    return {
        "t": 0,
        "m_u": torch.zeros(d, dtype=torch.float64),
        "s_u": torch.zeros(d, dtype=torch.float64),
        "m_v": torch.zeros(d, dtype=torch.float64),
        "s_v": torch.zeros(d, dtype=torch.float64),
    }


def step(
    u: torch.Tensor,
    v: torch.Tensor,
    grad_u: torch.Tensor,
    grad_v: torch.Tensor,
    state: dict[str, Any],
    hyperparameters: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Adam update step WITHOUT bias correction."""
    lr = float(hyperparameters["lr"])
    beta1 = float(hyperparameters["beta1"])
    beta2 = float(hyperparameters["beta2"])
    eps = float(hyperparameters["eps"])
    t = state["t"] + 1
    m_u = beta1 * state["m_u"] + (1.0 - beta1) * grad_u
    s_u = beta2 * state["s_u"] + (1.0 - beta2) * grad_u * grad_u
    u_new = u - lr * m_u / (torch.sqrt(s_u) + eps)
    m_v = beta1 * state["m_v"] + (1.0 - beta1) * grad_v
    s_v = beta2 * state["s_v"] + (1.0 - beta2) * grad_v * grad_v
    v_new = v - lr * m_v / (torch.sqrt(s_v) + eps)
    return u_new, v_new, {"t": t, "m_u": m_u, "s_u": s_u, "m_v": m_v, "s_v": s_v}
'''

OPS = [{"op": "replace", "file": _FILE, "start_line": 23, "end_line": 90, "content": _CONTENT}]
