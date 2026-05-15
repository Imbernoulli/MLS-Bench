"""Baseline: Original 3DGS densification (Kerbl et al., 2023).

Gradient-threshold based clone/split with periodic opacity reset.
"""

_FILE = "gsplat/custom_strategy.py"

_DEFAULT = '''
@dataclass
class CustomStrategy(Strategy):
    """Original 3DGS densification: clone small, split large, prune transparent."""

    prune_opa: float = 0.005
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    prune_scale3d: float = 0.1
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    reset_every: int = 3000
    refine_every: int = 100

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        return {"grad2d": None, "count": None, "scene_scale": scene_scale}

    def step_pre_backward(self, params, optimizers, state, step, info):
        info["means2d"].retain_grad()

    def step_post_backward(self, params, optimizers, state, step, info, packed=False):
        if step >= self.refine_stop_iter:
            return

        # Update gradient statistics
        grads = info["means2d"].grad.clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

        n = len(list(params.values())[0])
        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n, device=grads.device)
            state["count"] = torch.zeros(n, device=grads.device)

        sel = (info["radii"] > 0.0).all(dim=-1)
        gs_ids = torch.where(sel)[1]
        state["grad2d"].index_add_(0, gs_ids, grads[sel].norm(dim=-1))
        state["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32))

        if step > self.refine_start_iter and step % self.refine_every == 0:
            avg_grads = state["grad2d"] / state["count"].clamp_min(1)
            scene_scale = state["scene_scale"]

            # Clone small Gaussians with high gradient
            is_grad_high = avg_grads > self.grow_grad2d
            scale_max = torch.exp(params["scales"]).max(dim=-1).values
            is_small = scale_max <= self.grow_scale3d * scene_scale
            is_dupli = is_grad_high & is_small
            if is_dupli.sum() > 0:
                duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)

            # Split large Gaussians with high gradient
            is_split = is_grad_high & ~is_small
            is_split = torch.cat([is_split, torch.zeros(is_dupli.sum(), dtype=torch.bool, device=is_split.device)])
            if is_split.sum() > 0:
                split(params=params, optimizers=optimizers, state=state, mask=is_split)

            # Prune low-opacity and too-large Gaussians
            is_prune = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa
            if step > self.reset_every:
                scale_max = torch.exp(params["scales"]).max(dim=-1).values
                is_prune = is_prune | (scale_max > self.prune_scale3d * scene_scale)
            if is_prune.sum() > 0:
                remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

            state["grad2d"].zero_()
            state["count"].zero_()
            torch.cuda.empty_cache()

        # Periodic opacity reset
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
        "content": _DEFAULT,
    },
]
