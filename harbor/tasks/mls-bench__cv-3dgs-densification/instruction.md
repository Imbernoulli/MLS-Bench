# MLS-Bench: cv-3dgs-densification

# 3D Gaussian Splatting Densification Strategy

## Objective

Design a densification strategy for 3D Gaussian Splatting (3DGS) that improves
novel view synthesis quality on real-world scenes under a fixed training and
rendering pipeline.

## Background

3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023) represents scenes as
collections of anisotropic 3D Gaussians optimized via differentiable
rasterization. A central component of training is the **densification strategy**,
which controls how Gaussians are added, split, pruned, or otherwise reorganized
during optimization. Common operations include:

- **Clone** small Gaussians in under-reconstructed regions.
- **Split** large Gaussians into smaller ones to recover finer detail.
- **Prune** transparent or oversized Gaussians.
- **Reset** opacities periodically to encourage pruning of redundant Gaussians.

Recent work proposes various refinements:

- **AbsGS** (Ye et al., arXiv:2404.10484) — homodirectional view-space gradient
  using the absolute value of per-pixel sub-gradients to overcome
  over-reconstruction caused by gradient cancellation.
- **Mini-Splatting** (Fang & Wang, arXiv:2403.14166) — blur-aware splitting and
  importance-weighted stochastic sampling for Gaussian count control.
- **3DGS-MCMC** (Kheradmand et al., NeurIPS 2024 Spotlight, arXiv:2404.09591) —
  treats densification as Markov-Chain Monte Carlo sampling, replacing cloning
  with a relocation step that preserves the sampled distribution.
- **Taming-3DGS** (Mallick et al., SIGGRAPH Asia 2024, arXiv:2406.15643) —
  budgeted per-step densification controlled by maximum gradient blending.
- **EDC: Efficient Density Control** (Deng et al., arXiv:2411.10133) — long-axis
  splitting with explicit child-Gaussian opacity control plus recovery-aware
  pruning.

## Implementation Contract

Implement a `CustomStrategy` class in `custom_strategy.py`. The strategy
controls the full lifecycle of Gaussians during training via two hooks called
by the training loop:

```python
@dataclass
class CustomStrategy(Strategy):
    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        # Initialize running statistics for the strategy.
        ...

    def step_pre_backward(self, params, optimizers, state, step, info):
        # Called BEFORE loss.backward(). Use to retain gradients.
        ...

    def step_post_backward(self, params, optimizers, state, step, info, packed=False):
        # Called AFTER loss.backward() and optimizer.step().
        # Implement densification / pruning logic here.
        ...
```

### Available Operations (`gsplat.strategy.ops`)

- `duplicate(params, optimizers, state, mask)` — clone selected Gaussians.
- `split(params, optimizers, state, mask)` — split selected Gaussians (sample 2
  new positions from the covariance).
- `remove(params, optimizers, state, mask)` — remove selected Gaussians.
- `reset_opa(params, optimizers, state, value)` — reset all opacities to a
  value.
- `relocate(params, optimizers, state, mask, binoms, min_opacity)` — relocate
  dead Gaussians on top of live ones.
- `sample_add(params, optimizers, state, n, binoms, min_opacity)` — add new
  Gaussians sampled from the opacity distribution.
- `inject_noise_to_position(params, optimizers, state, scaler)` — perturb
  positions with Gaussian noise.

### Available Information

The `info` dict passed in by the rasterizer contains:

- `means2d` — 2D projected means (with `.grad` after backward).
- `width`, `height` — image dimensions.
- `n_cameras` — number of cameras in the batch.
- `radii` — screen-space radii per Gaussian.
- `gaussian_ids` — which Gaussians are visible.

The `params` dict contains:

- `means` — `[N, 3]` positions.
- `scales` — `[N, 3]` log-scales (use `torch.exp(...)` for actual scales).
- `quats` — `[N, 4]` rotation quaternions.
- `opacities` — `[N]` logit-opacities (use `torch.sigmoid(...)` for actual
  opacities).
- `sh0`, `shN` — spherical-harmonic colour coefficients.

### Fixed Pipeline

The renderer is the `gsplat` CUDA rasterizer. The full training and evaluation
pipeline (renderer, optimizer, photometric loss, schedule, and metrics) is fixed
by the harness and not editable. Your contribution must be confined to the
densification strategy in the editable region.

## Baselines

| Baseline   | Description |
|------------|-------------|
| `absgrad`  | gsplat `DefaultStrategy` with the AbsGS absolute-gradient criterion (Ye et al., arXiv:2404.10484). |
| `taming`   | Taming-3DGS budgeted densification with max-grad blending (Mallick et al., arXiv:2406.15643), combined with the AbsGS gradient and the revised opacity formula. |
| `edc`      | Taming densification combined with EDC long-axis splitting and recovery-aware pruning (Deng et al., arXiv:2411.10133). |

The contribution should be a transferable densification rule, not a change to the renderer, photometric loss, optimizer, or training pipeline.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/gsplat/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `gsplat/custom_strategy.py`
- editable lines **20–90**




## Readable Context


### `gsplat/custom_strategy.py`  [EDITABLE — lines 20–90 only]

```python
     1: """Custom densification strategy for 3D Gaussian Splatting.
     2: 
     3: This file defines the CustomStrategy class that controls how Gaussians
     4: are added, split, and pruned during per-scene optimization.
     5: """
     6: 
     7: from dataclasses import dataclass
     8: from typing import Any, Dict, Tuple, Union
     9: 
    10: import math
    11: import torch
    12: from gsplat.strategy.base import Strategy
    13: from gsplat.strategy.ops import (
    14:     duplicate, split, remove, reset_opa,
    15:     relocate, sample_add, inject_noise_to_position,
    16: )
    17: 
    18: 
    19: # ============================================================================
    20: # Densification Strategy (EDITABLE REGION: lines 20-90)
    21: # ============================================================================
    22: 
    23: @dataclass
    24: class CustomStrategy(Strategy):
    25:     """Custom 3DGS densification strategy.
    26: 
    27:     TODO: Design your densification strategy to maximize novel view quality.
    28: 
    29:     Available operations (from gsplat.strategy.ops):
    30:         duplicate(params, optimizers, state, mask)
    31:             Clone Gaussians selected by boolean mask.
    32:         split(params, optimizers, state, mask, revised_opacity=False)
    33:             Split Gaussians into 2 new ones sampled from covariance.
    34:         remove(params, optimizers, state, mask)
    35:             Remove Gaussians selected by boolean mask.
    36:         reset_opa(params, optimizers, state, value)
    37:             Reset all opacities to logit(value).
    38:         relocate(params, optimizers, state, mask, binoms, min_opacity)
    39:             Teleport dead Gaussians to high-opacity locations.
    40:         sample_add(params, optimizers, state, n, binoms, min_opacity)
    41:             Add n new Gaussians sampled from opacity distribution.
    42:         inject_noise_to_position(params, optimizers, state, scaler)
    43:             Perturb Gaussian positions with noise scaled by opacity.
    44: 
    45:     Available info from rasterization (in step_pre/post_backward):
    46:         info["means2d"]       - 2D projected means, call .retain_grad()
    47:         info["means2d"].grad  - gradient w.r.t. 2D means (after backward)
    48:         info["width"], info["height"] - image dimensions
    49:         info["n_cameras"]     - number of cameras in batch
    50:         info["radii"]         - screen-space radii [C, N, 1] or [C, N, 2]
    51:         info["gaussian_ids"]  - visible Gaussian indices
    52: 
    53:     Available params:
    54:         params["means"]      - [N, 3] world-space positions
    55:         params["scales"]     - [N, 3] log-scales (use torch.exp for actual)
    56:         params["quats"]      - [N, 4] rotation quaternions
    57:         params["opacities"]  - [N] logit-opacities (use torch.sigmoid)
    58:         params["sh0"]        - [N, 1, 3] DC spherical harmonics
    59:         params["shN"]        - [N, K, 3] higher-order SH coefficients
    60: 
    61:     Evaluation metrics (for reference):
    62:         PSNR (higher is better), SSIM (higher is better), LPIPS (lower is better)
    63:     """
    64: 
    65:     def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
    66:         """Initialize and return the running state for this strategy."""
    67:         raise NotImplementedError("Implement initialize_state")
    68: 
    69:     def check_sanity(self, params, optimizers):
    70:         """Sanity check for required parameters."""
    71:         super().check_sanity(params, optimizers)
    72:         for key in ["means", "scales", "quats", "opacities"]:
    73:             assert key in params, f"{key} is required in params but missing."
    74: 
    75:     def step_pre_backward(self, params, optimizers, state, step, info):
    76:         """Called BEFORE loss.backward(). Retain gradients for densification."""
    77:         raise NotImplementedError("Implement step_pre_backward")
    78: 
    79:     def step_post_backward(self, params, optimizers, state, step, info,
    80:                            packed=False):
    81:         """Called AFTER loss.backward(). Implement densification logic here."""
    82:         raise NotImplementedError("Implement step_post_backward")
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `absgrad` baseline — editable region  [READ-ONLY — reference implementation]

In `gsplat/custom_strategy.py`:

```python
Lines 20–91:
    17: 
    18: 
    19: # ============================================================================
    20: 
    21: @dataclass
    22: class CustomStrategy(Strategy):
    23:     """AbsGS: absolute gradient densification for fine detail recovery."""
    24: 
    25:     prune_opa: float = 0.005
    26:     grow_grad2d: float = 0.0006
    27:     grow_scale3d: float = 0.01
    28:     prune_scale3d: float = 0.1
    29:     refine_start_iter: int = 500
    30:     refine_stop_iter: int = 15_000
    31:     reset_every: int = 3000
    32:     refine_every: int = 100
    33: 
    34:     def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
    35:         return {"grad2d": None, "count": None, "scene_scale": scene_scale}
    36: 
    37:     def step_pre_backward(self, params, optimizers, state, step, info):
    38:         info["means2d"].retain_grad()
    39: 
    40:     def step_post_backward(self, params, optimizers, state, step, info, packed=False):
    41:         if step >= self.refine_stop_iter:
    42:             return
    43: 
    44:         if hasattr(info["means2d"], "absgrad"):
    45:             grads = info["means2d"].absgrad.clone()
    46:         else:
    47:             grads = info["means2d"].grad.abs().clone()
    48:         grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
    49:         grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
    50: 
    51:         n = len(list(params.values())[0])
    52:         if state["grad2d"] is None:
    53:             state["grad2d"] = torch.zeros(n, device=grads.device)
    54:             state["count"] = torch.zeros(n, device=grads.device)
    55: 
    56:         sel = (info["radii"] > 0.0).all(dim=-1)
    57:         gs_ids = torch.where(sel)[1]
    58:         state["grad2d"].index_add_(0, gs_ids, grads[sel].norm(dim=-1))
    59:         state["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32))
    60: 
    61:         if step > self.refine_start_iter and step % self.refine_every == 0:
    62:             avg_grads = state["grad2d"] / state["count"].clamp_min(1)
    63:             scene_scale = state["scene_scale"]
    64: 
    65:             is_grad_high = avg_grads > self.grow_grad2d
    66:             scale_max = torch.exp(params["scales"]).max(dim=-1).values
    67:             is_small = scale_max <= self.grow_scale3d * scene_scale
    68: 
    69:             is_dupli = is_grad_high & is_small
    70:             if is_dupli.sum() > 0:
    71:                 duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)
    72: 
    73:             is_split = is_grad_high & ~is_small
    74:             is_split = torch.cat([is_split, torch.zeros(is_dupli.sum(), dtype=torch.bool, device=is_split.device)])
    75:             if is_split.sum() > 0:
    76:                 split(params=params, optimizers=optimizers, state=state, mask=is_split)
    77: 
    78:             is_prune = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa
    79:             if step > self.reset_every:
    80:                 scale_max = torch.exp(params["scales"]).max(dim=-1).values
    81:                 is_prune = is_prune | (scale_max > self.prune_scale3d * scene_scale)
    82:             if is_prune.sum() > 0:
    83:                 remove(params=params, optimizers=optimizers, state=state, mask=is_prune)
    84: 
    85:             state["grad2d"].zero_()
    86:             state["count"].zero_()
    87:             torch.cuda.empty_cache()
    88: 
    89:         if step % self.reset_every == 0 and step > 0:
    90:             reset_opa(params=params, optimizers=optimizers, state=state,
    91:                       value=self.prune_opa * 2.0)
```

### `taming` baseline — editable region  [READ-ONLY — reference implementation]

In `gsplat/custom_strategy.py`:

```python
Lines 20–106:
    17: 
    18: 
    19: # ============================================================================
    20: 
    21: @dataclass
    22: class CustomStrategy(Strategy):
    23:     """AbsGS + Taming-3DGS (max-grad blend) + New Split (revised opacity)."""
    24: 
    25:     prune_opa: float = 0.005
    26:     grow_grad2d: float = 0.0005   # slightly lower than absgrad (more aggressive growth)
    27:     grow_scale3d: float = 0.01
    28:     prune_scale3d: float = 0.1
    29:     refine_start_iter: int = 500
    30:     refine_stop_iter: int = 18_000  # later stop — max-grad keeps finding splits
    31:     reset_every: int = 3000
    32:     refine_every: int = 100
    33:     # Taming-3DGS blend weights
    34:     avg_weight: float = 0.7
    35:     max_weight: float = 0.3
    36: 
    37:     def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
    38:         return {
    39:             "grad2d": None, "count": None, "grad2d_max": None,
    40:             "scene_scale": scene_scale,
    41:         }
    42: 
    43:     def step_pre_backward(self, params, optimizers, state, step, info):
    44:         info["means2d"].retain_grad()
    45: 
    46:     def step_post_backward(self, params, optimizers, state, step, info, packed=False):
    47:         if step >= self.refine_stop_iter:
    48:             return
    49: 
    50:         # AbsGS: absolute gradients (key vs. default)
    51:         if hasattr(info["means2d"], "absgrad"):
    52:             grads = info["means2d"].absgrad.clone()
    53:         else:
    54:             grads = info["means2d"].grad.abs().clone()
    55:         grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
    56:         grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
    57: 
    58:         n = len(list(params.values())[0])
    59:         if state["grad2d"] is None:
    60:             state["grad2d"] = torch.zeros(n, device=grads.device)
    61:             state["count"] = torch.zeros(n, device=grads.device)
    62:             state["grad2d_max"] = torch.zeros(n, device=grads.device)
    63: 
    64:         sel = (info["radii"] > 0.0).all(dim=-1)
    65:         gs_ids = torch.where(sel)[1]
    66:         grad_norms = grads[sel].norm(dim=-1)
    67:         state["grad2d"].index_add_(0, gs_ids, grad_norms)
    68:         state["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32))
    69:         # Taming-3DGS: track per-Gaussian max gradient (catches view-specific spikes)
    70:         state["grad2d_max"].scatter_reduce_(0, gs_ids, grad_norms, reduce="amax", include_self=True)
    71: 
    72:         if step > self.refine_start_iter and step % self.refine_every == 0:
    73:             avg_grads = state["grad2d"] / state["count"].clamp_min(1)
    74:             # Blended signal: avg for persistent errors, max for view-specific
    75:             combined = self.avg_weight * avg_grads + self.max_weight * state["grad2d_max"]
    76:             scene_scale = state["scene_scale"]
    77: 
    78:             is_grad_high = combined > self.grow_grad2d
    79:             scale_max = torch.exp(params["scales"]).max(dim=-1).values
    80:             is_small = scale_max <= self.grow_scale3d * scene_scale
    81: 
    82:             is_dupli = is_grad_high & is_small
    83:             if is_dupli.sum() > 0:
    84:                 duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)
    85: 
    86:             # New Split: revised_opacity=True preserves α-blending under splits
    87:             is_split = is_grad_high & ~is_small
    88:             is_split = torch.cat([is_split, torch.zeros(is_dupli.sum(), dtype=torch.bool, device=is_split.device)])
    89:             if is_split.sum() > 0:
    90:                 split(params=params, optimizers=optimizers, state=state, mask=is_split, revised_opacity=True)
    91: 
    92:             is_prune = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa
    93:             if step > self.reset_every:
    94:                 scale_max = torch.exp(params["scales"]).max(dim=-1).values
    95:                 is_prune = is_prune | (scale_max > self.prune_scale3d * scene_scale)
    96:             if is_prune.sum() > 0:
    97:                 remove(params=params, optimizers=optimizers, state=state, mask=is_prune)
    98: 
    99:             state["grad2d"].zero_()
   100:             state["count"].zero_()
   101:             state["grad2d_max"].zero_()
   102:             torch.cuda.empty_cache()
   103: 
   104:         if step % self.reset_every == 0 and step > 0:
   105:             reset_opa(params=params, optimizers=optimizers, state=state,
   106:                       value=self.prune_opa * 2.0)
```

### `edc` baseline — editable region  [READ-ONLY — reference implementation]

In `gsplat/custom_strategy.py`:

```python
Lines 20–177:
    17: 
    18: 
    19: # ============================================================================
    20: 
    21: @dataclass
    22: class CustomStrategy(Strategy):
    23:     """EDC-TamingGS-Abs: Long-Axis Split + Recovery-Aware Pruning + Taming + AbsGS."""
    24: 
    25:     prune_opa: float = 0.005
    26:     grow_grad2d: float = 0.0005
    27:     grow_scale3d: float = 0.01
    28:     prune_scale3d: float = 0.1
    29:     refine_start_iter: int = 500
    30:     refine_stop_iter: int = 22_000  # extended (recovery prune keeps count stable)
    31:     reset_every: int = 3000
    32:     refine_every: int = 100
    33:     # Taming-3DGS blend
    34:     avg_weight: float = 0.7
    35:     max_weight: float = 0.3
    36:     # EDC: Long-Axis Split
    37:     long_axis_opa_factor: float = 0.6   # child opacity = 0.6 · parent
    38:     long_axis_scale_div: float = 1.6    # longest axis scale shrunk by 1.6
    39:     long_axis_offset: float = 0.5       # child offset = ±0.5 · longest_axis
    40:     # EDC: Recovery-Aware Pruning
    41:     recovery_offset: int = 300          # iters after each opacity reset
    42:     recovery_opa: float = 0.05          # prune below this sigmoid-opacity
    43: 
    44:     def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
    45:         return {
    46:             "grad2d": None, "count": None, "grad2d_max": None,
    47:             "scene_scale": scene_scale,
    48:         }
    49: 
    50:     def step_pre_backward(self, params, optimizers, state, step, info):
    51:         info["means2d"].retain_grad()
    52: 
    53:     def _long_axis_split(self, params, optimizers, state, mask):
    54:         """EDC long-axis split: children placed deterministically along
    55:         longest axis, opacity = 0.6 · sigmoid(parent), longest axis / 1.6.
    56:         """
    57:         from gsplat.strategy.ops import _update_param_with_optimizer
    58:         from gsplat.utils import normalized_quat_to_rotmat
    59:         import torch.nn.functional as F
    60: 
    61:         sel = torch.where(mask)[0]
    62:         rest = torch.where(~mask)[0]
    63:         if len(sel) == 0:
    64:             return
    65: 
    66:         scales = torch.exp(params["scales"][sel])                  # [N, 3]
    67:         quats = F.normalize(params["quats"][sel], dim=-1)
    68:         rotmats = normalized_quat_to_rotmat(quats)                 # [N, 3, 3]
    69:         # longest axis index per Gaussian
    70:         max_axis = scales.argmax(dim=-1, keepdim=True)             # [N, 1]
    71:         # local one-hot direction along longest axis
    72:         e_local = torch.zeros_like(scales)
    73:         e_local.scatter_(1, max_axis, 1.0)                          # [N, 3]
    74:         # rotate to world frame
    75:         direction = torch.einsum("nij,nj->ni", rotmats, e_local)    # [N, 3]
    76:         longest = scales.gather(1, max_axis).squeeze(-1)            # [N]
    77:         # offsets ±0.5 · longest along world direction
    78:         offset = self.long_axis_offset * longest.unsqueeze(-1) * direction
    79:         samples = torch.stack([offset, -offset], dim=0)             # [2, N, 3]
    80: 
    81:         # new scales: longest axis / 1.6, others unchanged
    82:         new_scales = scales.clone()
    83:         new_scales.scatter_(1, max_axis, longest.unsqueeze(-1) / self.long_axis_scale_div)
    84: 
    85:         # new opacity: 0.6 · alpha, following the EDC long-axis split rule
    86:         new_opa_alpha = (self.long_axis_opa_factor * torch.sigmoid(params["opacities"][sel])).clamp(1e-6, 1.0 - 1e-6)
    87:         new_opa_logit = torch.logit(new_opa_alpha)
    88: 
    89:         def param_fn(name, p):
    90:             repeats = [2] + [1] * (p.dim() - 1)
    91:             if name == "means":
    92:                 p_split = (p[sel] + samples).reshape(-1, 3)
    93:             elif name == "scales":
    94:                 p_split = torch.log(new_scales).repeat(2, 1)
    95:             elif name == "opacities":
    96:                 p_split = new_opa_logit.repeat(repeats)
    97:             else:
    98:                 p_split = p[sel].repeat(repeats)
    99:             return torch.nn.Parameter(torch.cat([p[rest], p_split]), requires_grad=p.requires_grad)
   100: 
   101:         def optimizer_fn(key, v):
   102:             v_split = torch.zeros((2 * len(sel), *v.shape[1:]), device=v.device)
   103:             return torch.cat([v[rest], v_split])
   104: 
   105:         _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
   106:         for k, v in state.items():
   107:             if isinstance(v, torch.Tensor):
   108:                 repeats = [2] + [1] * (v.dim() - 1)
   109:                 state[k] = torch.cat((v[rest], v[sel].repeat(repeats)))
   110: 
   111:     def step_post_backward(self, params, optimizers, state, step, info, packed=False):
   112:         if step >= self.refine_stop_iter:
   113:             return
   114: 
   115:         # AbsGS: absolute gradients
   116:         if hasattr(info["means2d"], "absgrad"):
   117:             grads = info["means2d"].absgrad.clone()
   118:         else:
   119:             grads = info["means2d"].grad.abs().clone()
   120:         grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
   121:         grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
   122: 
   123:         n = len(list(params.values())[0])
   124:         if state["grad2d"] is None:
   125:             state["grad2d"] = torch.zeros(n, device=grads.device)
   126:             state["count"] = torch.zeros(n, device=grads.device)
   127:             state["grad2d_max"] = torch.zeros(n, device=grads.device)
   128: 
   129:         sel = (info["radii"] > 0.0).all(dim=-1)
   130:         gs_ids = torch.where(sel)[1]
   131:         grad_norms = grads[sel].norm(dim=-1)
   132:         state["grad2d"].index_add_(0, gs_ids, grad_norms)
   133:         state["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32))
   134:         # Taming: per-Gaussian max gradient
   135:         state["grad2d_max"].scatter_reduce_(0, gs_ids, grad_norms, reduce="amax", include_self=True)
   136: 
   137:         # EDC Recovery-Aware Pruning: triggered 300 iters after each opacity reset (after first reset)
   138:         if step > self.reset_every and (step - self.recovery_offset) % self.reset_every == 0:
   139:             opa = torch.sigmoid(params["opacities"].flatten())
   140:             is_recovery_prune = opa < self.recovery_opa
   141:             if is_recovery_prune.sum() > 0:
   142:                 remove(params=params, optimizers=optimizers, state=state, mask=is_recovery_prune)
   143: 
   144:         if step > self.refine_start_iter and step % self.refine_every == 0:
   145:             avg_grads = state["grad2d"] / state["count"].clamp_min(1)
   146:             combined = self.avg_weight * avg_grads + self.max_weight * state["grad2d_max"]
   147:             scene_scale = state["scene_scale"]
   148: 
   149:             is_grad_high = combined > self.grow_grad2d
   150:             scale_max = torch.exp(params["scales"]).max(dim=-1).values
   151:             is_small = scale_max <= self.grow_scale3d * scene_scale
   152: 
   153:             is_dupli = is_grad_high & is_small
   154:             if is_dupli.sum() > 0:
   155:                 duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)
   156: 
   157:             # EDC long-axis split (replaces stochastic split)
   158:             is_split = is_grad_high & ~is_small
   159:             is_split = torch.cat([is_split, torch.zeros(is_dupli.sum(), dtype=torch.bool, device=is_split.device)])
   160:             if is_split.sum() > 0:
   161:                 self._long_axis_split(params, optimizers, state, is_split)
   162: 
   163:             is_prune = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa
   164:             if step > self.reset_every:
   165:                 scale_max = torch.exp(params["scales"]).max(dim=-1).values
   166:                 is_prune = is_prune | (scale_max > self.prune_scale3d * scene_scale)
   167:             if is_prune.sum() > 0:
   168:                 remove(params=params, optimizers=optimizers, state=state, mask=is_prune)
   169: 
   170:             state["grad2d"].zero_()
   171:             state["count"].zero_()
   172:             state["grad2d_max"].zero_()
   173:             torch.cuda.empty_cache()
   174: 
   175:         if step % self.reset_every == 0 and step > 0:
   176:             reset_opa(params=params, optimizers=optimizers, state=state,
   177:                       value=self.prune_opa * 2.0)
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
