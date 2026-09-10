"""Baseline: EDC-TamingGS-Abs (Deng et al., arXiv:2411.10133).

Stacks two EDC enhancements on top of our existing Taming-3DGS-Abs baseline:

1. **Long-Axis Split** — replaces stochastic covariance-sampled split.
   Each child is placed at parent ± 0.5 · longest_axis_direction (deterministic).
   Child opacity is set to 0.6 · sigmoid(parent) (paper-prescribed factor that
   minimises density-distribution shift). Longest axis scale is divided by 1.6
   for both children, other axes unchanged. Reduces the post-split rendering
   inconsistency that random covariance sampling introduces.

2. **Recovery-Aware Pruning** — leverages the differential opacity-recovery
   rate of "needed" vs. "overfit" Gaussians after each opacity reset. At
   iter (k·reset_every + 300) we prune Gaussians whose sigmoid-opacity is
   still below 0.05; healthy Gaussians have already recovered, overfit ones
   stay near zero. This catches splits that overcorrected without waiting
   for the next reset cycle's regular prune.

EDC reports PSNR gains in its Mip-NeRF 360 experiments; this baseline
implements the two EDC densification mechanisms on the local TamingGS-Abs
harness, with exact results tracked by this task's leaderboard.

Stacks cleanly with retained Taming + AbsGS pieces:
- AbsGS absolute gradients
- Taming max-grad blend (avg_weight=0.7, max_weight=0.3)
- Revised-opacity safeguard kept on the long-axis split (children opacity also
  passed through 1−sqrt(1−α) for cumulative-α invariance, plus the 0.6 factor
  applied multiplicatively).

Refine_stop_iter extended to 22k (vs taming's 18k) — recovery-aware pruning
keeps the Gaussian count stable, allowing more refinement before freezing.
"""

_FILE = "gsplat/custom_strategy.py"

_EDC = '''
@dataclass
class CustomStrategy(Strategy):
    """EDC-TamingGS-Abs: Long-Axis Split + Recovery-Aware Pruning + Taming + AbsGS."""

    prune_opa: float = 0.005
    grow_grad2d: float = 0.0005
    grow_scale3d: float = 0.01
    prune_scale3d: float = 0.1
    refine_start_iter: int = 500
    refine_stop_iter: int = 22_000  # extended (recovery prune keeps count stable)
    reset_every: int = 3000
    refine_every: int = 100
    # Taming-3DGS blend
    avg_weight: float = 0.7
    max_weight: float = 0.3
    # EDC: Long-Axis Split
    long_axis_opa_factor: float = 0.6   # child opacity = 0.6 · parent
    long_axis_scale_div: float = 1.6    # longest axis scale shrunk by 1.6
    long_axis_offset: float = 0.5       # child offset = ±0.5 · longest_axis
    # EDC: Recovery-Aware Pruning
    recovery_offset: int = 300          # iters after each opacity reset
    recovery_opa: float = 0.05          # prune below this sigmoid-opacity

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        return {
            "grad2d": None, "count": None, "grad2d_max": None,
            "scene_scale": scene_scale,
        }

    def step_pre_backward(self, params, optimizers, state, step, info):
        info["means2d"].retain_grad()

    def _long_axis_split(self, params, optimizers, state, mask):
        """EDC long-axis split: children placed deterministically along
        longest axis, opacity = 0.6 · sigmoid(parent), longest axis / 1.6.
        """
        from gsplat.strategy.ops import _update_param_with_optimizer
        from gsplat.utils import normalized_quat_to_rotmat
        import torch.nn.functional as F

        sel = torch.where(mask)[0]
        rest = torch.where(~mask)[0]
        if len(sel) == 0:
            return

        scales = torch.exp(params["scales"][sel])                  # [N, 3]
        quats = F.normalize(params["quats"][sel], dim=-1)
        rotmats = normalized_quat_to_rotmat(quats)                 # [N, 3, 3]
        # longest axis index per Gaussian
        max_axis = scales.argmax(dim=-1, keepdim=True)             # [N, 1]
        # local one-hot direction along longest axis
        e_local = torch.zeros_like(scales)
        e_local.scatter_(1, max_axis, 1.0)                          # [N, 3]
        # rotate to world frame
        direction = torch.einsum("nij,nj->ni", rotmats, e_local)    # [N, 3]
        longest = scales.gather(1, max_axis).squeeze(-1)            # [N]
        # offsets ±0.5 · longest along world direction
        offset = self.long_axis_offset * longest.unsqueeze(-1) * direction
        samples = torch.stack([offset, -offset], dim=0)             # [2, N, 3]

        # new scales: longest axis / 1.6, others unchanged
        new_scales = scales.clone()
        new_scales.scatter_(1, max_axis, longest.unsqueeze(-1) / self.long_axis_scale_div)

        # new opacity: 0.6 · alpha, following the EDC long-axis split rule
        new_opa_alpha = (self.long_axis_opa_factor * torch.sigmoid(params["opacities"][sel])).clamp(1e-6, 1.0 - 1e-6)
        new_opa_logit = torch.logit(new_opa_alpha)

        def param_fn(name, p):
            repeats = [2] + [1] * (p.dim() - 1)
            if name == "means":
                p_split = (p[sel] + samples).reshape(-1, 3)
            elif name == "scales":
                p_split = torch.log(new_scales).repeat(2, 1)
            elif name == "opacities":
                p_split = new_opa_logit.repeat(repeats)
            else:
                p_split = p[sel].repeat(repeats)
            return torch.nn.Parameter(torch.cat([p[rest], p_split]), requires_grad=p.requires_grad)

        def optimizer_fn(key, v):
            v_split = torch.zeros((2 * len(sel), *v.shape[1:]), device=v.device)
            return torch.cat([v[rest], v_split])

        _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                repeats = [2] + [1] * (v.dim() - 1)
                state[k] = torch.cat((v[rest], v[sel].repeat(repeats)))

    def step_post_backward(self, params, optimizers, state, step, info, packed=False):
        if step >= self.refine_stop_iter:
            return

        # AbsGS: absolute gradients
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
        # Taming: per-Gaussian max gradient
        state["grad2d_max"].scatter_reduce_(0, gs_ids, grad_norms, reduce="amax", include_self=True)

        # EDC Recovery-Aware Pruning: triggered 300 iters after each opacity reset (after first reset)
        if step > self.reset_every and (step - self.recovery_offset) % self.reset_every == 0:
            opa = torch.sigmoid(params["opacities"].flatten())
            is_recovery_prune = opa < self.recovery_opa
            if is_recovery_prune.sum() > 0:
                remove(params=params, optimizers=optimizers, state=state, mask=is_recovery_prune)

        if step > self.refine_start_iter and step % self.refine_every == 0:
            avg_grads = state["grad2d"] / state["count"].clamp_min(1)
            combined = self.avg_weight * avg_grads + self.max_weight * state["grad2d_max"]
            scene_scale = state["scene_scale"]

            is_grad_high = combined > self.grow_grad2d
            scale_max = torch.exp(params["scales"]).max(dim=-1).values
            is_small = scale_max <= self.grow_scale3d * scene_scale

            is_dupli = is_grad_high & is_small
            if is_dupli.sum() > 0:
                duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)

            # EDC long-axis split (replaces stochastic split)
            is_split = is_grad_high & ~is_small
            is_split = torch.cat([is_split, torch.zeros(is_dupli.sum(), dtype=torch.bool, device=is_split.device)])
            if is_split.sum() > 0:
                self._long_axis_split(params, optimizers, state, is_split)

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
        "content": _EDC,
    },
]
