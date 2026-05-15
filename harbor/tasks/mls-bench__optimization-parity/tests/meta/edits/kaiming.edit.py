"""Kaiming initialization baseline for optimization-parity.

Uses Kaiming normal initialization (He init) instead of Xavier uniform,
paired with a moderately-sized dataset and tuned AdamW hyperparameters
(lower weight decay, slightly higher learning rate).
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
    secret: tuple[int, ...],
    config: TaskConfig,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a moderately-sized random dataset (50k examples)."""
    generator = torch.Generator().manual_seed(seed)
    num_examples = min(50_000, config.max_train_examples)
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
        "start_line": 220,
        "end_line": 255,
        "content": _CONTENT,
    },
]
