# MLS-Bench: security-adversarial-attack-sparse-l0

# Sparse Adversarial Attack (L0 Constraint)

## Research Question
Can you design a stronger sparse adversarial attack that fools image classifiers by perturbing only a small number of spatial pixels?

## Background
Sparse adversarial attacks differ from dense `L_p` attacks in that the perturbation is restricted in `L0` rather than `L_inf` or `L2`: only a handful of input pixels may be modified, but each modified pixel can change by an arbitrary amount within `[0, 1]`. The sparsity constraint matches threat models such as physical patches, image-tag manipulation, and pixel-level corruption, and it is also informative because gradient-based attacks tend to spread perturbations across many pixels and are not well suited to it.

Representative sparse-attack algorithms include the Jacobian Saliency Map Attack JSMA (Papernot et al., 2016, arXiv:1511.07528), the differential-evolution One-Pixel attack (Su et al., 2019, arXiv:1710.08864), the geometry-inspired SparseFool (Modas et al., CVPR 2019, arXiv:1811.02248), the random-search Sparse-RS framework (Croce et al., AAAI 2022, arXiv:2006.12834), and Pixle, a fast pixel-rearrangement black-box attack (Pomponi et al., 2022, arXiv:2202.02236).

## Objective
Implement a stronger sparse attack in `bench/custom_attack.py`. The method should maximize attack success rate (ASR) under a strict `L0` perturbation budget:

- Threat model: full model access for custom attack implementation (gradients permitted).
- Norm constraint: number of modified spatial pixels is bounded.
- Budget: `L0(x_adv, x) <= pixels`, where `pixels = 10`. A pixel is counted as modified if any of its channels changes.

## Editable Interface
You must implement:

`run_attack(model, images, labels, pixels, device, n_classes) -> adv_images`

Inputs:
- `images`: tensor of shape `(N, C, H, W)`, values in `[0, 1]`.
- `labels`: tensor of shape `(N,)`.
- `pixels`: maximum number of modified spatial pixels per sample.
- `n_classes`: 10 for CIFAR-10, 100 for CIFAR-100.

Output:
- `adv_images`: same shape as `images`, also in `[0, 1]`.

## Evaluation Protocol
Each evaluation script:
1. Loads one pretrained model.
2. Collects up to 1000 samples that are initially classified correctly.
3. Runs your `run_attack`.
4. Checks `L0` validity (`<= pixels` modified spatial pixels) and `[0, 1]` range.
5. Reports `clean_acc`, `robust_acc`, and `asr = 1 - robust_acc`.

Important:
- ASR denominator is the number of initially correct samples.
- Invalid adversarial outputs (shape mismatch, non-finite values, or violated budget) are treated as failure.

## Evaluation Scenarios
Each scenario is a (model, dataset) pair drawn from {ResNet20, VGG11-BN, MobileNetV2} x {CIFAR-10, CIFAR-100}, using publicly available pretrained checkpoints.

## Baselines
The baselines below run inside the same harness via edit ops; reference implementations are in `torchattacks`:

- `onepixel`: One-Pixel attack (Su et al., 2019, arXiv:1710.08864). Differential-evolution sparse attack with default population and iteration settings from the paper.
- `sparsefool`: SparseFool (Modas et al., CVPR 2019, arXiv:1811.02248). Geometry-inspired sparse attack using DeepFool-like linearization.
- `jsma`: JSMA (Papernot et al., 2016, arXiv:1511.07528). Jacobian saliency map-based targeted sparse attack.
- `pixle`: Pixle (Pomponi et al., 2022, arXiv:2202.02236). Pixel-rearrangement-based black-box sparse attack.
- `sparse_rs`: Sparse-RS (Croce et al., AAAI 2022, arXiv:2006.12834). Random-search L0 attack from https://github.com/fra31/sparse-rs.

The goal is to improve ASR while respecting the L0 budget.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/torchattacks/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to score zero.

- `torchattacks/bench/custom_attack.py`
- editable lines **3–26**


Other files you may **read** for context (do not modify):
- `torchattacks/bench/run_eval.py`
- `torchattacks/torchattacks/attacks/_differential_evolution.py`


## Readable Context


### `torchattacks/bench/custom_attack.py`  [EDITABLE — lines 3–26 only]

```python
     1: import torch
     2: import torch.nn as nn
     3: 
     4: # =====================================================================
     5: # EDITABLE: implement run_attack below
     6: # =====================================================================
     7: def run_attack(
     8:     model: nn.Module,
     9:     images: torch.Tensor,
    10:     labels: torch.Tensor,
    11:     pixels: int,
    12:     device: torch.device,
    13:     n_classes: int,
    14: ) -> torch.Tensor:
    15:     """
    16:     Sparse L0 adversarial attack.
    17:     images: (N, C, H, W) in [0, 1] on device. labels: (N,) on device.
    18:     pixels: max number of modified spatial pixels (H, W) per sample.
    19:     n_classes: 10 for CIFAR-10, 100 for CIFAR-100.
    20:     Returns adv_images satisfying an L0 pixel budget validated by evaluator.
    21:     """
    22:     _ = (model, labels, pixels, device, n_classes)
    23:     return images.clone()
    24: 
    25: # =====================================================================
    26: # END EDITABLE REGION
    27: # =====================================================================
```




## How You Will Be Evaluated

After you finish, evaluation runs a fixed set of scripts and aggregates the
metrics they emit. These scripts are **not** in your workspace — you cannot
read or modify them. The labels below indicate what each evaluation tests:

- **ResNet20-C10** — wall-clock budget `1:00:00`, compute share `0.5`
- **VGG11BN-C10** — wall-clock budget `1:00:00`, compute share `0.5`
- **MobileNetV2-C10** — wall-clock budget `1:00:00`, compute share `0.5`
- **ResNet20-C100** — wall-clock budget `1:00:00`, compute share `0.5`
- **MobileNetV2-C100** — wall-clock budget `1:00:00`, compute share `0.5`


Scoring uses the same `combined_score` aggregation as the MLS-Bench
leaderboard. Multiple seeds are averaged.



## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `onepixel` baseline — editable region  [READ-ONLY — reference implementation]

In `torchattacks/bench/custom_attack.py`:

```python
Lines 3–29:
     1: import torch
     2: import torch.nn as nn
     3: 
     4: # =====================================================================
     5: # EDITABLE: implement run_attack below
     6: # =====================================================================
     7: def run_attack(
     8:     model: nn.Module,
     9:     images: torch.Tensor,
    10:     labels: torch.Tensor,
    11:     pixels: int,
    12:     device: torch.device,
    13:     n_classes: int,
    14: ) -> torch.Tensor:
    15:     import torchattacks
    16: 
    17:     _ = (device, n_classes)
    18:     model.eval()
    19:     attack = torchattacks.OnePixel(
    20:         model,
    21:         pixels=pixels,
    22:         steps=6,
    23:         popsize=8,
    24:         inf_batch=128,
    25:     )
    26:     return attack(images, labels)
    27: 
    28: # =====================================================================
    29: # END EDITABLE REGION
    30: # =====================================================================
```

### `sparsefool` baseline — editable region  [READ-ONLY — reference implementation]

In `torchattacks/bench/custom_attack.py`:

```python
Lines 3–28:
     1: import torch
     2: import torch.nn as nn
     3: 
     4: # =====================================================================
     5: # EDITABLE: implement run_attack below
     6: # =====================================================================
     7: def run_attack(
     8:     model: nn.Module,
     9:     images: torch.Tensor,
    10:     labels: torch.Tensor,
    11:     pixels: int,
    12:     device: torch.device,
    13:     n_classes: int,
    14: ) -> torch.Tensor:
    15:     import torchattacks
    16: 
    17:     _ = (pixels, device, n_classes)
    18:     model.eval()
    19:     attack = torchattacks.SparseFool(
    20:         model,
    21:         steps=20,
    22:         lam=3.0,
    23:         overshoot=0.02,
    24:     )
    25:     return attack(images, labels)
    26: 
    27: # =====================================================================
    28: # END EDITABLE REGION
    29: # =====================================================================
```

### `jsma` baseline — editable region  [READ-ONLY — reference implementation]

In `torchattacks/bench/custom_attack.py`:

```python
Lines 3–32:
     1: import torch
     2: import torch.nn as nn
     3: 
     4: # =====================================================================
     5: # EDITABLE: implement run_attack below
     6: # =====================================================================
     7: def run_attack(
     8:     model: nn.Module,
     9:     images: torch.Tensor,
    10:     labels: torch.Tensor,
    11:     pixels: int,
    12:     device: torch.device,
    13:     n_classes: int,
    14: ) -> torch.Tensor:
    15:     import torchattacks
    16: 
    17:     _ = (device, n_classes)
    18:     model.eval()
    19: 
    20:     # gamma bounds total perturbed features (C*H*W space) to `pixels`, which
    21:     # is a sufficient upper bound on the number of distinct spatial pixels.
    22:     num_features = int(images.shape[1] * images.shape[2] * images.shape[3])
    23:     gamma = float(pixels) / float(num_features)
    24: 
    25:     attack = torchattacks.JSMA(model, theta=1.0, gamma=gamma)
    26:     # Least-likely class as target -> strong untargeted proxy, works for any
    27:     # n_classes (fixes CIFAR-100 out-of-range target bug).
    28:     attack.set_mode_targeted_least_likely(quiet=True)
    29:     return attack(images, labels)
    30: 
    31: # =====================================================================
    32: # END EDITABLE REGION
    33: # =====================================================================
```

### `pixle` baseline — editable region  [READ-ONLY — reference implementation]

In `torchattacks/bench/custom_attack.py`:

```python
Lines 3–31:
     1: import torch
     2: import torch.nn as nn
     3: 
     4: # =====================================================================
     5: # EDITABLE: implement run_attack below
     6: # =====================================================================
     7: def run_attack(
     8:     model: nn.Module,
     9:     images: torch.Tensor,
    10:     labels: torch.Tensor,
    11:     pixels: int,
    12:     device: torch.device,
    13:     n_classes: int,
    14: ) -> torch.Tensor:
    15:     import torchattacks
    16: 
    17:     _ = (pixels, device, n_classes)
    18:     model.eval()
    19:     attack = torchattacks.Pixle(
    20:         model,
    21:         x_dimensions=(1, 2),
    22:         y_dimensions=(1, 2),
    23:         pixel_mapping="random",
    24:         restarts=3,
    25:         max_iterations=5,
    26:         update_each_iteration=False,
    27:     )
    28:     return attack(images, labels)
    29: 
    30: # =====================================================================
    31: # END EDITABLE REGION
    32: # =====================================================================
```

### `sparse_rs` baseline — editable region  [READ-ONLY — reference implementation]

In `torchattacks/bench/custom_attack.py`:

```python
Lines 3–147:
     1: import torch
     2: import torch.nn as nn
     3: 
     4: # =====================================================================
     5: # EDITABLE: implement run_attack below
     6: # =====================================================================
     7: def run_attack(
     8:     model: nn.Module,
     9:     images: torch.Tensor,
    10:     labels: torch.Tensor,
    11:     pixels: int,
    12:     device: torch.device,
    13:     n_classes: int,
    14: ) -> torch.Tensor:
    15:     """Sparse-RS L0 black-box attack (Croce et al., AAAI 2022)."""
    16:     import torch
    17:     import torch.nn.functional as F
    18: 
    19:     _ = (n_classes,)
    20:     model.eval()
    21: 
    22:     n_queries = 1000
    23:     p_init = 0.8
    24:     eps = int(pixels)
    25: 
    26:     x = images.detach().clone().to(device)
    27:     y = labels.detach().clone().to(device)
    28:     B, C, H, W = x.shape
    29:     n_pixels = H * W
    30: 
    31:     def _margin_and_loss(xb, yb):
    32:         with torch.no_grad():
    33:             logits = model(xb)
    34:         u = torch.arange(xb.shape[0], device=xb.device)
    35:         y_corr = logits[u, yb].clone()
    36:         logits[u, yb] = -float("inf")
    37:         y_others = logits.max(dim=-1)[0]
    38:         margin = y_corr - y_others
    39:         return margin, margin  # 'margin' loss variant
    40: 
    41:     def _p_selection(it):
    42:         # Rescaled schedule (see Sparse-RS paper Fig. 3 / rs_attacks.py).
    43:         it = int(it / n_queries * 10000)
    44:         if 0 < it <= 50:
    45:             return p_init / 2
    46:         if 50 < it <= 200:
    47:             return p_init / 4
    48:         if 200 < it <= 500:
    49:             return p_init / 5
    50:         if 500 < it <= 1000:
    51:             return p_init / 6
    52:         if 1000 < it <= 2000:
    53:             return p_init / 8
    54:         if 2000 < it <= 4000:
    55:             return p_init / 10
    56:         if 4000 < it <= 6000:
    57:             return p_init / 12
    58:         if 6000 < it <= 8000:
    59:             return p_init / 15
    60:         if 8000 < it:
    61:             return p_init / 20
    62:         return p_init
    63: 
    64:     def _rand_colors(shape):
    65:         # Binary {0,1} random colors, as in Sparse-RS default.
    66:         return torch.randint(0, 2, shape, device=device, dtype=x.dtype)
    67: 
    68:     # ---- Initialise: random eps pixels per image with random binary colors.
    69:     x_best = x.clone()
    70:     b_all = torch.zeros(B, eps, dtype=torch.long, device=device)
    71:     be_all = torch.zeros(B, n_pixels - eps, dtype=torch.long, device=device)
    72:     for i in range(B):
    73:         perm = torch.randperm(n_pixels, device=device)
    74:         ind_p = perm[:eps]
    75:         ind_np = perm[eps:]
    76:         x_best[i, :, ind_p // W, ind_p % W] = _rand_colors((C, eps)).clamp(0.0, 1.0)
    77:         b_all[i] = ind_p
    78:         be_all[i] = ind_np
    79: 
    80:     margin_min, loss_min = _margin_and_loss(x_best, y)
    81: 
    82:     for it in range(1, n_queries):
    83:         idx_to_fool = (margin_min > 0.0).nonzero().squeeze(-1)
    84:         if idx_to_fool.numel() == 0:
    85:             break
    86: 
    87:         x_curr = x[idx_to_fool].clone()
    88:         x_best_curr = x_best[idx_to_fool].clone()
    89:         y_curr = y[idx_to_fool]
    90:         margin_curr = margin_min[idx_to_fool].clone()
    91:         loss_curr = loss_min[idx_to_fool].clone()
    92:         b_curr = b_all[idx_to_fool].clone()
    93:         be_curr = be_all[idx_to_fool].clone()
    94: 
    95:         x_new = x_best_curr.clone()
    96:         eps_it = max(int(_p_selection(it) * eps), 1)
    97:         ind_p = torch.randperm(eps, device=device)[:eps_it]
    98:         ind_np = torch.randperm(n_pixels - eps, device=device)[:eps_it]
    99: 
   100:         for i in range(x_new.shape[0]):
   101:             p_set = b_curr[i, ind_p]
   102:             np_set = be_curr[i, ind_np]
   103:             # Restore previously-perturbed positions to clean.
   104:             x_new[i, :, p_set // W, p_set % W] = x_curr[i, :, p_set // W, p_set % W]
   105:             # Perturb newly-selected positions with random binary colors.
   106:             if eps_it > 1:
   107:                 x_new[i, :, np_set // W, np_set % W] = _rand_colors((C, eps_it)).clamp(0.0, 1.0)
   108:             else:
   109:                 old = x_new[i, :, np_set // W, np_set % W].clone()
   110:                 new = old.clone()
   111:                 tries = 0
   112:                 while (new == old).all() and tries < 16:
   113:                     new = _rand_colors((C, 1)).clamp(0.0, 1.0)
   114:                     tries += 1
   115:                 x_new[i, :, np_set // W, np_set % W] = new
   116: 
   117:         margin, loss = _margin_and_loss(x_new, y_curr)
   118: 
   119:         idx_improved = (loss < loss_curr).float()
   120:         idx_miscl = (margin < -1e-6).float()
   121:         idx_keep = torch.max(idx_improved, idx_miscl)
   122:         nkeep = int(idx_keep.sum().item())
   123: 
   124:         # Update loss whenever loss improves.
   125:         upd_loss = (idx_improved > 0).nonzero().squeeze(-1)
   126:         if upd_loss.numel() > 0:
   127:             loss_min[idx_to_fool[upd_loss]] = loss[upd_loss]
   128: 
   129:         if nkeep > 0:
   130:             upd = (idx_keep > 0).nonzero().squeeze(-1)
   131:             margin_min[idx_to_fool[upd]] = margin[upd]
   132:             x_best[idx_to_fool[upd]] = x_new[upd]
   133: 
   134:             # Swap mask indices for the accepted moves.
   135:             # `upd` comes from .squeeze(-1), so the batch dim is preserved
   136:             # (shape [K] with K>=1); always use the 2-D batched form.
   137:             t = b_curr[upd].clone()
   138:             te = be_curr[upd].clone()
   139:             t[:, ind_p] = be_curr[upd][:, ind_np]
   140:             te[:, ind_np] = b_curr[upd][:, ind_p]
   141:             b_all[idx_to_fool[upd]] = t
   142:             be_all[idx_to_fool[upd]] = te
   143: 
   144:     return x_best.detach()
   145: 
   146: # =====================================================================
   147: # END EDITABLE REGION
   148: # =====================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
