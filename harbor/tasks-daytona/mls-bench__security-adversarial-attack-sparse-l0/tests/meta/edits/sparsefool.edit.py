"""SparseFool baseline for security-adversarial-attack-sparse-l0."""

_FILE = "torchattacks/bench/custom_attack.py"

_SPARSEFOOL_FN = """\
def run_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    pixels: int,
    device: torch.device,
    n_classes: int,
) -> torch.Tensor:
    import torchattacks

    _ = (pixels, device, n_classes)
    model.eval()
    attack = torchattacks.SparseFool(
        model,
        steps=20,
        lam=3.0,
        overshoot=0.02,
    )
    return attack(images, labels)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 7,
        "end_line": 23,
        "content": _SPARSEFOOL_FN,
    }
]
