"""Baseline: Taming-3DGS (Mallick et al., 2024) + revised split opacity (Rota Bulo et al., 2024).

Combines three research-backed enhancements that stack cleanly:

1. **AbsGS** (Ye et al., 2024, arXiv:2404.10484) — absolute gradients
   capture magnitude regardless of sign-cancellation, recovering fine
   detail that avg-gradient densification misses.

2. **Taming-3DGS** (Mallick et al., 2024, arXiv:2406.15643) — track the
   **per-Gaussian max gradient** across the accumulation window in
   addition to the mean. The blended signal `0.7·avg + 0.3·max` catches
   both persistent errors (avg) and view-specific spikes (max), which a
   pure avg-gradient criterion misses when a Gaussian only fails in a
   small number of views.

3. **New Split** (Rota Bulo et al., ECCV 2024 "Revising Densification in
   Gaussian Splatting") — mathematically consistent splitting via the
   `revised_opacity=True` flag preserves cumulative α-blending under
   splits: each child opacity = 1 − sqrt(1 − α_parent), so compound
   rendering stays invariant. Without this, splits silently brighten
   regions because raw copies double the effective opacity.

Extends `refine_stop_iter` to 18k (vs. 15k default) — max-grad tracking
keeps finding useful split candidates longer than pure avg-grad does.
"""

_FILE = "gsplat/custom_strategy.py"

_TAMING = '''
@dataclass
class CustomStrategy(Strategy):
    """AbsGS + Taming-3DGS (max-grad blend) + New Split (revised opacity)."""

    prune_opa: float = 0.005
    grow_grad2d: float = 0.0005   # slightly lower than absgrad (more aggressive growth)
    grow_scale3d: float = 0.01
    prune_scale3d: float = 0.1
    refine_start_iter: int = 500
    refine_stop_iter: int = 18_000  # later stop — max-grad keeps finding splits
    reset_every: int = 3000
    refine_every: int = 100
    # Taming-3DGS blend weights
    avg_weight: float = 0.7
    max_weight: float = 0.3

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        return {
            "grad2d": None, "count": None, "grad2d_max": None,
            "scene_scale": scene_scale,
        }

    def step_pre_backward(self, params, optimizers, state, step, info):
        info["means2d"].retain_grad()

    def step_post_backward(self, params, optimizers, state, step, info, packed=False):
        if step >= self.refine_stop_iter:
            return

        # AbsGS: absolute gradients (key vs. default)
        if hasattr(info["means2d"], "absgrad"):
            grads = info["means2d"].absgrad.clone()
        else:
            grads = info["means2d"].grad.abs().clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

        n = len(list(params.values())[0])
        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n, device=grads.device)
            state["count"] = torch.zeros(n, device=grads.device)
            state["grad2d_max"] = torch.zeros(n, device=grads.device)

        sel = (info["radii"] > 0.0).all(dim=-1)
        gs_ids = torch.where(sel)[1]
        grad_norms = grads[sel].norm(dim=-1)
        state["grad2d"].index_add_(0, gs_ids, grad_norms)
        state["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32))
        # Taming-3DGS: track per-Gaussian max gradient (catches view-specific spikes)
        state["grad2d_max"].scatter_reduce_(0, gs_ids, grad_norms, reduce="amax", include_self=True)

        if step > self.refine_start_iter and step % self.refine_every == 0:
            avg_grads = state["grad2d"] / state["count"].clamp_min(1)
            # Blended signal: avg for persistent errors, max for view-specific
            combined = self.avg_weight * avg_grads + self.max_weight * state["grad2d_max"]
            scene_scale = state["scene_scale"]

            is_grad_high = combined > self.grow_grad2d
            scale_max = torch.exp(params["scales"]).max(dim=-1).values
            is_small = scale_max <= self.grow_scale3d * scene_scale

            is_dupli = is_grad_high & is_small
            if is_dupli.sum() > 0:
                duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)

            # New Split: revised_opacity=True preserves α-blending under splits
            is_split = is_grad_high & ~is_small
            is_split = torch.cat([is_split, torch.zeros(is_dupli.sum(), dtype=torch.bool, device=is_split.device)])
            if is_split.sum() > 0:
                split(params=params, optimizers=optimizers, state=state, mask=is_split, revised_opacity=True)

            is_prune = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa
            if step > self.reset_every:
                scale_max = torch.exp(params["scales"]).max(dim=-1).values
                is_prune = is_prune | (scale_max > self.prune_scale3d * scene_scale)
            if is_prune.sum() > 0:
                remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

            state["grad2d"].zero_()
            state["count"].zero_()
            state["grad2d_max"].zero_()
            torch.cuda.empty_cache()

        if step % self.reset_every == 0 and step > 0:
            reset_opa(params=params, optimizers=optimizers, state=state,
                      value=self.prune_opa * 2.0)
'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 20,
        "end_line": 90,
        "content": _TAMING,
    },
]
