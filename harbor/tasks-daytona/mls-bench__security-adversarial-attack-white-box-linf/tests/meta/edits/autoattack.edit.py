"""AutoAttack baseline for security-adversarial-attack-white-box-linf."""

_FILE = "torchattacks/bench/custom_attack.py"

_AUTOATTACK_FN = """\
def run_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float,
    device: torch.device,
    n_classes: int,
) -> torch.Tensor:
    import os
    import torchattacks

    _ = device
    model.eval()
    attack = torchattacks.AutoAttack(
        model,
        norm="Linf",
        eps=eps,
        version="standard",
        n_classes=n_classes,
        seed=int(os.environ.get("SEED", "42")),
        verbose=False,
    )
    return attack(images, labels)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 7,
        "end_line": 22,
        "content": _AUTOATTACK_FN,
    }
]
