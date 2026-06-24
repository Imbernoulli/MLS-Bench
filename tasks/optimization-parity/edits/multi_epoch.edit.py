"""Multi-epoch baseline for optimization-parity.

Compared with the naive one-pass baseline, this variant uses a smaller,
configurable slice of the labeled pool (default 10_000). This intentionally
causes repeated passes over the same samples under the fixed step budget while
keeping standard initialization and AdamW defaults.
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
    """Use a smaller, configurable slice of the labeled pool for multi-epoch reuse."""
    train_examples = 10_000  # Tunable parameter for this multi-epoch baseline.
    num_examples = min(train_examples, config.max_train_examples)
    return x_pool[:num_examples], y_pool[:num_examples]


def get_optimizer_config(config: TaskConfig) -> dict[str, float]:
    """Return AdamW hyperparameters for the fixed training loop."""
    return {
        "lr": 1e-3,
        "wd": 1e-2,
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
