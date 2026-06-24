"""Kaiming initialization baseline for optimization-parity.

Uses Kaiming normal initialization (He init) instead of Xavier uniform,
paired with a moderately-sized slice of the labeled pool and tuned AdamW
hyperparameters (lower weight decay, slightly higher learning rate).
"""

_FILE = "pytorch-examples/optimization_parity/custom_strategy.py"

_CONTENT = '''\
def init_model(model: nn.Sequential, config: TaskConfig) -> None:
    """Initialize the fixed two-layer MLP with Kaiming normal initialization."""
    for layer in model:
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)


def make_dataset(
    x_pool: torch.Tensor,
    y_pool: torch.Tensor,
    config: TaskConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a moderately-sized slice of the labeled pool (50k examples)."""
    num_examples = min(50_000, config.max_train_examples)
    return x_pool[:num_examples], y_pool[:num_examples]


def get_optimizer_config(config: TaskConfig) -> dict[str, float]:
    """Return tuned AdamW hyperparameters with lower weight decay."""
    return {
        "lr": 2e-3,
        "wd": 1e-3,
        "beta1": 0.9,
        "beta2": 0.999,
    }
'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 242,
        "end_line": 274,
        "content": _CONTENT,
    },
]
