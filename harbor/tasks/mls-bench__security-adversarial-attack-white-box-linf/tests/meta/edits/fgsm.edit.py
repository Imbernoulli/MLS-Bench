"""FGSM baseline for security-adversarial-attack-white-box-linf."""

_FILE = "torchattacks/bench/custom_attack.py"

_FGSM_FN = """\
def run_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float,
    device: torch.device,
    n_classes: int,
) -> torch.Tensor:
    import torch.nn.functional as F

    _ = (device, n_classes)
    model.eval()
    x = images.detach().clone().requires_grad_(True)
    logits = model(x)
    loss = F.cross_entropy(logits, labels)
    grad = torch.autograd.grad(loss, x)[0]

    with torch.no_grad():
        x_adv = x + eps * grad.sign()
        delta = torch.clamp(x_adv - images, min=-eps, max=eps)
        x_adv = torch.clamp(images + delta, 0.0, 1.0)

    return x_adv.detach()
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 7,
        "end_line": 22,
        "content": _FGSM_FN,
    }
]
