"""SGD baseline (lr=0.05) for opt-diagonal-net."""

_FILE = "RAIN/opt_diagonal_net/custom_optimizer.py"

_CONTENT = '''\
def get_hyperparameters(
    dim: int,
    sparsity: int,
    delta: float,
) -> dict[str, Any]:
    """SGD hyperparameters: lr=0.05."""
    return {"lr": 0.05}


def init_state(
    u: torch.Tensor,
    v: torch.Tensor,
    hyperparameters: dict[str, Any],
) -> dict[str, Any]:
    """SGD requires no additional state."""
    return {"t": 0}


def step(
    u: torch.Tensor,
    v: torch.Tensor,
    grad_u: torch.Tensor,
    grad_v: torch.Tensor,
    state: dict[str, Any],
    hyperparameters: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Vanilla gradient descent step."""
    lr = float(hyperparameters["lr"])
    state["t"] = state.get("t", 0) + 1
    return u - lr * grad_u, v - lr * grad_v, state
'''

OPS = [{"op": "replace", "file": _FILE, "start_line": 23, "end_line": 90, "content": _CONTENT}]
