"""Random-search baseline for security-adversarial-attack-black-box-score."""

_FILE = "torchattacks/bench/custom_attack.py"

_RANDOM_FN = """\
def run_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float,
    n_queries: int,
    device: torch.device,
    n_classes: int,
) -> torch.Tensor:
    _ = (device, n_classes)
    model.eval()

    adv_images = images.detach().clone()
    step = eps / 2.0
    n_steps = max(1, min(int(n_queries), 64))

    with torch.no_grad():
        best = model(adv_images).gather(1, labels.view(-1, 1)).squeeze(1)

        for _ in range(n_steps):
            noise = torch.empty_like(adv_images).uniform_(-step, step)
            cand = adv_images + noise
            cand = torch.clamp(images + torch.clamp(cand - images, -eps, eps), 0.0, 1.0)

            cand_score = model(cand).gather(1, labels.view(-1, 1)).squeeze(1)
            improve = cand_score < best

            if improve.any():
                mask = improve.view(-1, 1, 1, 1)
                adv_images = torch.where(mask, cand, adv_images)
                best = torch.where(improve, cand_score, best)

    delta = torch.clamp(adv_images - images, min=-eps, max=eps)
    return torch.clamp(images + delta, 0.0, 1.0).detach()
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 7,
        "end_line": 56,
        "content": _RANDOM_FN,
    }
]
