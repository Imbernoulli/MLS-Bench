"""OnePixel baseline for security-adversarial-attack-sparse-l0."""

_FILE = "torchattacks/bench/custom_attack.py"

_ONEPIXEL_FN = """\
def run_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    pixels: int,
    device: torch.device,
    n_classes: int,
) -> torch.Tensor:
    import torchattacks

    _ = (device, n_classes)
    model.eval()
    attack = torchattacks.OnePixel(
        model,
        pixels=pixels,
        steps=6,
        popsize=8,
        inf_batch=128,
    )
    return attack(images, labels)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 7,
        "end_line": 23,
        "content": _ONEPIXEL_FN,
    }
]
