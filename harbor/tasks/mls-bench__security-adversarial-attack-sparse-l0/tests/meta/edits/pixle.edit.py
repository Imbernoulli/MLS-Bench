"""Pixle baseline for security-adversarial-attack-sparse-l0."""

_FILE = "torchattacks/bench/custom_attack.py"

_PIXLE_FN = """\
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
    attack = torchattacks.Pixle(
        model,
        x_dimensions=(1, 2),
        y_dimensions=(1, 2),
        pixel_mapping="random",
        restarts=3,
        max_iterations=5,
        update_each_iteration=False,
    )
    return attack(images, labels)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 7,
        "end_line": 23,
        "content": _PIXLE_FN,
    }
]
