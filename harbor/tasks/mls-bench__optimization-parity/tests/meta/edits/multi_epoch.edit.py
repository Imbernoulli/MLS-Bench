"""Multi-epoch baseline for optimization-parity.

Compared with the naive one-pass baseline, this variant uses a smaller,
configurable training set size (default 10_000). This intentionally causes repeated passes
over the same samples under the fixed step budget while keeping standard
initialization and AdamW defaults.
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
    """Use a smaller, configurable random dataset to allow multi-epoch reuse."""
    generator = torch.Generator().manual_seed(seed)
    train_examples = 10_000  # Tunable parameter for this multi-epoch baseline.
    num_examples = min(train_examples, config.max_train_examples)

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
        "start_line": 220,
        "end_line": 255,
        "content": _CONTENT,
    },
]
