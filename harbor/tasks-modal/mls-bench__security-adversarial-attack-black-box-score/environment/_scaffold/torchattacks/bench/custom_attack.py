import torch
import torch.nn as nn

# =====================================================================
# EDITABLE: implement run_attack below
# =====================================================================
def run_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float,
    n_queries: int,
    device: torch.device,
    n_classes: int,
) -> torch.Tensor:
    """
    Score-based query black-box attack under Linf constraint.

    Args:
        model: black-box wrapper that only exposes forward logits.
        images: (N, C, H, W) in [0, 1], on device.
        labels: (N,), on device.
        eps: Linf budget.
        n_queries: per-sample query budget.
        device: runtime device.
        n_classes: number of classes.
    """
    _ = (device, n_classes)
    model.eval()

    # A simple default that already performs score-based search.
    # Baselines will replace this block with stronger algorithms.
    adv = images.detach().clone()
    step = eps / 4.0
    iters = max(1, min(int(n_queries), 16))

    with torch.no_grad():
        for _ in range(iters):
            logits_old = model(adv)
            true_old = logits_old.gather(1, labels.view(-1, 1)).squeeze(1)

            noise = torch.empty_like(adv).uniform_(-step, step)
            cand = adv + noise
            cand = torch.clamp(images + torch.clamp(cand - images, -eps, eps), 0.0, 1.0)

            logits_new = model(cand)
            true_new = logits_new.gather(1, labels.view(-1, 1)).squeeze(1)
            improve = true_new < true_old

            if improve.any():
                mask = improve.view(-1, 1, 1, 1)
                adv = torch.where(mask, cand, adv)

    delta = torch.clamp(adv - images, min=-eps, max=eps)
    adv = torch.clamp(images + delta, 0.0, 1.0)
    return adv.detach()

# =====================================================================
# END EDITABLE REGION
# =====================================================================
