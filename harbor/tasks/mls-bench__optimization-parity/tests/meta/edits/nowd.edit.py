"""No weight decay baseline for optimization-parity.

Same as the default baseline but with weight_decay set to 0.
Tests whether weight decay helps or hurts on sparse parity learning.
"""

_FILE = "pytorch-examples/optimization_parity/custom_strategy.py"

_CONTENT = '''\
def init_model(model: nn.Sequential, config: TaskConfig) -> None:
    """Initialize the fixed two-layer MLP without using the hidden secret."""
    for layer in model:
        if isinstance(layer, nn.Linear):
            gain = nn.init.calculate_gain("relu") if layer is model[0] else 1.0
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)


def make_dataset(
    secret: tuple[int, ...],
    config: TaskConfig,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a maximal random dataset to induce one-pass training."""
    generator = torch.Generator().manual_seed(seed)
    num_examples = config.max_train_examples
    x = torch.randint(
        low=0,
        high=2,
        size=(num_examples, config.n_features),
        generator=generator,
        dtype=torch.int64,
    ).to(torch.float32)
    y = parity_labels(x, secret)
    return x, y


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
        "start_line": 220,
        "end_line": 255,
        "content": _CONTENT,
    },
]
