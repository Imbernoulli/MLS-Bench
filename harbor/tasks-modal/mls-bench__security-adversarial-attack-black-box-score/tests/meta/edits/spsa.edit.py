"""SPSA baseline for security-adversarial-attack-black-box-score."""

_FILE = "torchattacks/bench/custom_attack.py"

_SPSA_FN = """\
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

    nb_sample = 128
    nb_iter = max(1, int(n_queries) // (2 * nb_sample))

    attack = torchattacks.SPSA(
        model=model,
        eps=eps,
        delta=0.01,
        lr=0.01,
        nb_iter=nb_iter,
        nb_sample=nb_sample,
        max_batch_size=64,
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
        "content": _SPSA_FN,
    }
]
