"""Square baseline for security-adversarial-attack-black-box-score."""

_FILE = "torchattacks/bench/custom_attack.py"

_SQUARE_FN = """\
def run_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float,
    n_queries: int,
    device: torch.device,
    n_classes: int,
) -> torch.Tensor:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    import torchattacks

    _ = (device, n_classes)
    model.eval()

    attack = torchattacks.Square(
        model=model,
        norm="Linf",
        eps=eps,
        n_queries=max(1, int(n_queries)),
        n_restarts=1,
        p_init=0.8,
        seed=int(os.environ.get("SEED", "42")),
        verbose=False,
        loss="margin",
        resc_schedule=True,
    )
    adv_images = attack(images, labels)
    delta = torch.clamp(adv_images - images, min=-eps, max=eps)
    return torch.clamp(images + delta, 0.0, 1.0).detach()
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 7,
        "end_line": 56,
        "content": _SQUARE_FN,
    }
]
