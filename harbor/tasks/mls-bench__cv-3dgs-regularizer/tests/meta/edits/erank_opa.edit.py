"""Baseline: erank + full-strength scale_opa (no anisotropy).

Combines two known-effective mechanisms at unmodified strengths:

  1. **scale_opa** (3DGS-MCMC) at coefficient 1e-2 each — same as the
     stand-alone scale_opa baseline. L1 on exp(scales) and sigmoid(opa)
     for compactness/sparsity.

  2. **erank log-barrier** (Hyung et al., NeurIPS 2024, arXiv:2406.11672)
     at coefficient 1e-2, applied after step 7000 — same warmup as the
     stand-alone erank baseline. Pushes effective rank ≥ 2 (planar
     Gaussians) to suppress needle-floater artifacts.

The stand-alone erank baseline uses HALF-strength scale_opa (5e-3 each)
to avoid the log-barrier blowing up. Here we use full 1e-2 scale_opa
because the additional compactness pressure helps indoor scenes (bonsai)
where erank alone underperforms.

Drops the anisotropy term that the earlier `triple` baseline included —
that term over-regularised stump.
"""

_FILE = "gsplat/custom_regularizer.py"

_ERANK_OPA = '''
# scale_opa (full strength) + erank log-barrier (warmup at step 7000).
SCALE_REG = 1e-2
OPACITY_REG = 1e-2
ERANK_REG = 1e-2
ERANK_WARMUP = 7000
ERANK_EPS = 1e-5

def compute_regularizer(splats, step, scene_scale):
    """Compactness L1 (always on) + erank log-barrier (after warmup)."""
    s = torch.exp(splats["scales"])                                # [N, 3]
    a = torch.sigmoid(splats["opacities"])                         # [N]

    loss = SCALE_REG * s.mean() + OPACITY_REG * a.mean()

    if step >= ERANK_WARMUP:
        s_sq = s * s
        q = s_sq / (s_sq.sum(dim=-1, keepdim=True) + 1e-12)
        H = -(q * (q + 1e-12).log()).sum(dim=-1)
        erank = H.exp()
        barrier = torch.clamp(-torch.log(erank - 1.0 + ERANK_EPS), min=0.0)
        s_min = s.min(dim=-1).values
        loss = loss + ERANK_REG * (barrier.mean() + s_min.mean())

    return loss
'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 37,
        "end_line": 51,
        "content": _ERANK_OPA,
    },
]
