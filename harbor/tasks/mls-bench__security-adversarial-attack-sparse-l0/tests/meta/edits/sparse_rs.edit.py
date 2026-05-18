"""Sparse-RS baseline for security-adversarial-attack-sparse-l0.

Implements the L0 branch of Sparse-RS from Croce et al. AAAI 2022
"Sparse-RS: a versatile framework for query-efficient sparse black-box
adversarial attacks" (https://arxiv.org/abs/2006.12834), reference code
https://github.com/fra31/sparse-rs/blob/master/rs_attacks.py.

Sparse-RS is the current SOTA for L0 (sparse) adversarial attacks. It is a
random-search black-box algorithm that maintains an L0 mask of size ``eps``
(== pixel budget) and at each iteration swaps a fraction of perturbed/clean
pixels and redraws random colors, keeping the move only if the margin loss
improves. It natively operates on spatial pixels, matching this benchmark's
L0 definition (``changed_pixels`` is counted via ``.any(dim=1)`` across
channels in bench/run_eval.py).

torchattacks has no Sparse_RS implementation (verified 2026-04 against
https://github.com/Harry24k/adversarial-attacks-pytorch/tree/master/torchattacks/attacks),
so we inline the L0 attack here.
"""

_FILE = "torchattacks/bench/custom_attack.py"

_SPARSE_RS_FN = r'''
def run_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    pixels: int,
    device: torch.device,
    n_classes: int,
) -> torch.Tensor:
    """Sparse-RS L0 black-box attack (Croce et al., AAAI 2022)."""
    import torch
    import torch.nn.functional as F

    _ = (n_classes,)
    model.eval()

    n_queries = 1000
    p_init = 0.8
    eps = int(pixels)

    x = images.detach().clone().to(device)
    y = labels.detach().clone().to(device)
    B, C, H, W = x.shape
    n_pixels = H * W

    def _margin_and_loss(xb, yb):
        with torch.no_grad():
            logits = model(xb)
        u = torch.arange(xb.shape[0], device=xb.device)
        y_corr = logits[u, yb].clone()
        logits[u, yb] = -float("inf")
        y_others = logits.max(dim=-1)[0]
        margin = y_corr - y_others
        return margin, margin  # 'margin' loss variant

    def _p_selection(it):
        # Rescaled schedule (see Sparse-RS paper Fig. 3 / rs_attacks.py).
        it = int(it / n_queries * 10000)
        if 0 < it <= 50:
            return p_init / 2
        if 50 < it <= 200:
            return p_init / 4
        if 200 < it <= 500:
            return p_init / 5
        if 500 < it <= 1000:
            return p_init / 6
        if 1000 < it <= 2000:
            return p_init / 8
        if 2000 < it <= 4000:
            return p_init / 10
        if 4000 < it <= 6000:
            return p_init / 12
        if 6000 < it <= 8000:
            return p_init / 15
        if 8000 < it:
            return p_init / 20
        return p_init

    def _rand_colors(shape):
        # Binary {0,1} random colors, as in Sparse-RS default.
        return torch.randint(0, 2, shape, device=device, dtype=x.dtype)

    # ---- Initialise: random eps pixels per image with random binary colors.
    x_best = x.clone()
    b_all = torch.zeros(B, eps, dtype=torch.long, device=device)
    be_all = torch.zeros(B, n_pixels - eps, dtype=torch.long, device=device)
    for i in range(B):
        perm = torch.randperm(n_pixels, device=device)
        ind_p = perm[:eps]
        ind_np = perm[eps:]
        x_best[i, :, ind_p // W, ind_p % W] = _rand_colors((C, eps)).clamp(0.0, 1.0)
        b_all[i] = ind_p
        be_all[i] = ind_np

    margin_min, loss_min = _margin_and_loss(x_best, y)

    for it in range(1, n_queries):
        idx_to_fool = (margin_min > 0.0).nonzero().squeeze(-1)
        if idx_to_fool.numel() == 0:
            break

        x_curr = x[idx_to_fool].clone()
        x_best_curr = x_best[idx_to_fool].clone()
        y_curr = y[idx_to_fool]
        margin_curr = margin_min[idx_to_fool].clone()
        loss_curr = loss_min[idx_to_fool].clone()
        b_curr = b_all[idx_to_fool].clone()
        be_curr = be_all[idx_to_fool].clone()

        x_new = x_best_curr.clone()
        eps_it = max(int(_p_selection(it) * eps), 1)
        ind_p = torch.randperm(eps, device=device)[:eps_it]
        ind_np = torch.randperm(n_pixels - eps, device=device)[:eps_it]

        for i in range(x_new.shape[0]):
            p_set = b_curr[i, ind_p]
            np_set = be_curr[i, ind_np]
            # Restore previously-perturbed positions to clean.
            x_new[i, :, p_set // W, p_set % W] = x_curr[i, :, p_set // W, p_set % W]
            # Perturb newly-selected positions with random binary colors.
            if eps_it > 1:
                x_new[i, :, np_set // W, np_set % W] = _rand_colors((C, eps_it)).clamp(0.0, 1.0)
            else:
                old = x_new[i, :, np_set // W, np_set % W].clone()
                new = old.clone()
                tries = 0
                while (new == old).all() and tries < 16:
                    new = _rand_colors((C, 1)).clamp(0.0, 1.0)
                    tries += 1
                x_new[i, :, np_set // W, np_set % W] = new

        margin, loss = _margin_and_loss(x_new, y_curr)

        idx_improved = (loss < loss_curr).float()
        idx_miscl = (margin < -1e-6).float()
        idx_keep = torch.max(idx_improved, idx_miscl)
        nkeep = int(idx_keep.sum().item())

        # Update loss whenever loss improves.
        upd_loss = (idx_improved > 0).nonzero().squeeze(-1)
        if upd_loss.numel() > 0:
            loss_min[idx_to_fool[upd_loss]] = loss[upd_loss]

        if nkeep > 0:
            upd = (idx_keep > 0).nonzero().squeeze(-1)
            margin_min[idx_to_fool[upd]] = margin[upd]
            x_best[idx_to_fool[upd]] = x_new[upd]

            # Swap mask indices for the accepted moves.
            # `upd` comes from .squeeze(-1), so the batch dim is preserved
            # (shape [K] with K>=1); always use the 2-D batched form.
            t = b_curr[upd].clone()
            te = be_curr[upd].clone()
            t[:, ind_p] = be_curr[upd][:, ind_np]
            te[:, ind_np] = b_curr[upd][:, ind_p]
            b_all[idx_to_fool[upd]] = t
            be_all[idx_to_fool[upd]] = te

    return x_best.detach()
'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 7,
        "end_line": 23,
        "content": _SPARSE_RS_FN.lstrip("\n"),
    }
]
