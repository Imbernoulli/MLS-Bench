"""MI-FGSM baseline for security-adversarial-attack-white-box-linf."""

_FILE = "torchattacks/bench/custom_attack.py"

_MIFGSM_FN = """\
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
    steps = 40
    alpha = eps / 10.0
    decay = 1.0

    x = images.detach()
    x_adv = x + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
    momentum = torch.zeros_like(x)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, labels)
        grad = torch.autograd.grad(loss, x_adv)[0]
        grad = grad / (grad.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-12)
        momentum = decay * momentum + grad

        with torch.no_grad():
            x_adv = x_adv + alpha * momentum.sign()
            delta = torch.clamp(x_adv - x, min=-eps, max=eps)
            x_adv = torch.clamp(x + delta, 0.0, 1.0)

        x_adv = x_adv.detach()

    return x_adv
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 7,
        "end_line": 22,
        "content": _MIFGSM_FN,
    }
]
