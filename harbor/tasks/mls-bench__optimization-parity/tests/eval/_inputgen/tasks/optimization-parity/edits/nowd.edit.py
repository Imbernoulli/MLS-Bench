"""No weight decay baseline for optimization-parity.

Same as the default baseline but with weight_decay set to 0.
Tests whether weight decay helps or hurts on sparse parity learning.
"""

_FILE = "pytorch-examples/optimization_parity/custom_strategy.py"

_CONTENT = '''\
def init_model(model: nn.Sequential, config: TaskConfig) -> None:
    """Initialize the fixed two-layer MLP."""
    for layer in model:
        if isinstance(layer, nn.Linear):
            gain = nn.init.calculate_gain("relu") if layer is model[0] else 1.0
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)


def make_dataset(
    x_pool: torch.Tensor,
    y_pool: torch.Tensor,
    config: TaskConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the maximal slice of the labeled pool to induce one-pass training."""
    num_examples = config.max_train_examples
    return x_pool[:num_examples], y_pool[:num_examples]


def get_optimizer_config(config: TaskConfig) -> dict[str, float]:
    """Return AdamW hyperparameters with no weight decay."""
    return {
        "lr": 1e-3,
        "wd": 0.0,
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
