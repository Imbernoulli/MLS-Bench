# MLS-Bench: security-adversarial-attack-black-box-score

# Score-Based Query Black-Box Attack under Linf Constraint

## Research Question
Can you design a stronger score-based query black-box attack under a fixed query budget and `L_inf` perturbation constraint?

## Background
Score-based query black-box attacks assume the attacker can query the victim model and observe its logits (or softmax scores) but cannot access gradients or weights. The attacker must search the input space using only forward queries while staying inside an `L_inf` ball around the clean image. This regime models realistic threat scenarios such as MLOps APIs that expose only prediction confidences.

Representative algorithms include the random-search-based Square Attack (Andriushchenko et al., 2020, arXiv:1912.00049), gradient-free SPSA-based attacks (Uesato et al., 2018, arXiv:1802.05666), and pixel-coordinate random search baselines. Across these methods the central tradeoff is between per-step exploration (which helps escape local minima) and per-step exploitation (which keeps the query budget low).

## Objective
Implement a better black-box attack in `bench/custom_attack.py`:

- Threat model: query black-box (no gradient access, only model logits).
- Constraint: `||x_adv - x||_inf <= eps`.
- Budget: `n_queries` is a per-sample query budget.

## Editable Interface
You must implement:

`run_attack(model, images, labels, eps, n_queries, device, n_classes) -> adv_images`

Inputs:
- `model`: black-box wrapper that returns logits only.
- `images`: tensor of shape `(N, C, H, W)`, in `[0, 1]`.
- `labels`: tensor of shape `(N,)`.
- `n_classes`: number of classes in the target dataset.

Output:
- `adv_images`: tensor with same shape as `images`, values in `[0, 1]`.

## Trusted Evaluation Logic
The evaluation logic in `bench/run_eval.py` is fixed and not editable.

- It tracks all model queries through a wrapper.
- If a batch exceeds query budget (`batch_size * n_queries`), the entire batch is marked as attack failure.
- `L_inf` and `[0, 1]` validity are checked per sample; only invalid samples are marked as attack failure.

Wrapper behavior and evaluation logic are fixed. Improvements should be confined to the attack algorithm in `custom_attack.py`.

## Query Semantics
- One call to `model(x)` consumes `x.shape[0]` queries.
- Repeated calls on the same sample still consume additional queries.
- Different batch partitioning is treated as equivalent total budget usage.

## Evaluation Scenarios
The evaluation is run across multiple (model, dataset) pairs using publicly available pretrained image classifiers.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/torchattacks/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `torchattacks/bench/custom_attack.py`
- editable lines **7–56**


Other files you may **read** for context (do not modify):
- `torchattacks/bench/run_eval.py`


## Readable Context


### `torchattacks/bench/custom_attack.py`  [EDITABLE — lines 7–56 only]

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
    11:     eps: float,
    12:     n_queries: int,
    13:     device: torch.device,
    14:     n_classes: int,
    15: ) -> torch.Tensor:
    16:     """
    17:     Score-based query black-box attack under Linf constraint.
    18: 
    19:     Args:
    20:         model: black-box wrapper that only exposes forward logits.
    21:         images: (N, C, H, W) in [0, 1], on device.
    22:         labels: (N,), on device.
    23:         eps: Linf budget.
    24:         n_queries: per-sample query budget.
    25:         device: runtime device.
    26:         n_classes: number of classes.
    27:     """
    28:     _ = (device, n_classes)
    29:     model.eval()
    30: 
    31:     # A simple default that already performs score-based search.
    32:     # Baselines will replace this block with stronger algorithms.
    33:     adv = images.detach().clone()
    34:     step = eps / 4.0
    35:     iters = max(1, min(int(n_queries), 16))
    36: 
    37:     with torch.no_grad():
    38:         for _ in range(iters):
    39:             logits_old = model(adv)
    40:             true_old = logits_old.gather(1, labels.view(-1, 1)).squeeze(1)
    41: 
    42:             noise = torch.empty_like(adv).uniform_(-step, step)
    43:             cand = adv + noise
    44:             cand = torch.clamp(images + torch.clamp(cand - images, -eps, eps), 0.0, 1.0)
    45: 
    46:             logits_new = model(cand)
    47:             true_new = logits_new.gather(1, labels.view(-1, 1)).squeeze(1)
    48:             improve = true_new < true_old
    49: 
    50:             if improve.any():
    51:                 mask = improve.view(-1, 1, 1, 1)
    52:                 adv = torch.where(mask, cand, adv)
    53: 
    54:     delta = torch.clamp(adv - images, min=-eps, max=eps)
    55:     adv = torch.clamp(images + delta, 0.0, 1.0)
    56:     return adv.detach()
    57: 
    58: # =====================================================================
    59: # END EDITABLE REGION
    60: # =====================================================================
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `square` baseline — editable region  [READ-ONLY — reference implementation]

In `torchattacks/bench/custom_attack.py`:

```python
Lines 7–38:
     4: # =====================================================================
     5: # EDITABLE: implement run_attack below
     6: # =====================================================================
     7: def run_attack(
     8:     model: nn.Module,
     9:     images: torch.Tensor,
    10:     labels: torch.Tensor,
    11:     eps: float,
    12:     n_queries: int,
    13:     device: torch.device,
    14:     n_classes: int,
    15: ) -> torch.Tensor:
    16:     import sys
    17:     import os
    18:     sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    19:     import torchattacks
    20: 
    21:     _ = (device, n_classes)
    22:     model.eval()
    23: 
    24:     attack = torchattacks.Square(
    25:         model=model,
    26:         norm="Linf",
    27:         eps=eps,
    28:         n_queries=max(1, int(n_queries)),
    29:         n_restarts=1,
    30:         p_init=0.8,
    31:         seed=int(os.environ.get("SEED", "42")),
    32:         verbose=False,
    33:         loss="margin",
    34:         resc_schedule=True,
    35:     )
    36:     adv_images = attack(images, labels)
    37:     delta = torch.clamp(adv_images - images, min=-eps, max=eps)
    38:     return torch.clamp(images + delta, 0.0, 1.0).detach()
    39: 
    40: # =====================================================================
    41: # END EDITABLE REGION
```

### `spsa` baseline — editable region  [READ-ONLY — reference implementation]

In `torchattacks/bench/custom_attack.py`:

```python
Lines 7–38:
     4: # =====================================================================
     5: # EDITABLE: implement run_attack below
     6: # =====================================================================
     7: def run_attack(
     8:     model: nn.Module,
     9:     images: torch.Tensor,
    10:     labels: torch.Tensor,
    11:     eps: float,
    12:     n_queries: int,
    13:     device: torch.device,
    14:     n_classes: int,
    15: ) -> torch.Tensor:
    16:     import sys
    17:     import os
    18:     sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    19:     import torchattacks
    20: 
    21:     _ = (device, n_classes)
    22:     model.eval()
    23: 
    24:     nb_sample = 128
    25:     nb_iter = max(1, int(n_queries) // (2 * nb_sample))
    26: 
    27:     attack = torchattacks.SPSA(
    28:         model=model,
    29:         eps=eps,
    30:         delta=0.01,
    31:         lr=0.01,
    32:         nb_iter=nb_iter,
    33:         nb_sample=nb_sample,
    34:         max_batch_size=64,
    35:     )
    36:     adv_images = attack(images, labels)
    37:     delta = torch.clamp(adv_images - images, min=-eps, max=eps)
    38:     return torch.clamp(images + delta, 0.0, 1.0).detach()
    39: 
    40: # =====================================================================
    41: # END EDITABLE REGION
```

### `random_search` baseline — editable region  [READ-ONLY — reference implementation]

In `torchattacks/bench/custom_attack.py`:

```python
Lines 7–40:
     4: # =====================================================================
     5: # EDITABLE: implement run_attack below
     6: # =====================================================================
     7: def run_attack(
     8:     model: nn.Module,
     9:     images: torch.Tensor,
    10:     labels: torch.Tensor,
    11:     eps: float,
    12:     n_queries: int,
    13:     device: torch.device,
    14:     n_classes: int,
    15: ) -> torch.Tensor:
    16:     _ = (device, n_classes)
    17:     model.eval()
    18: 
    19:     adv_images = images.detach().clone()
    20:     step = eps / 2.0
    21:     n_steps = max(1, min(int(n_queries), 64))
    22: 
    23:     with torch.no_grad():
    24:         best = model(adv_images).gather(1, labels.view(-1, 1)).squeeze(1)
    25: 
    26:         for _ in range(n_steps):
    27:             noise = torch.empty_like(adv_images).uniform_(-step, step)
    28:             cand = adv_images + noise
    29:             cand = torch.clamp(images + torch.clamp(cand - images, -eps, eps), 0.0, 1.0)
    30: 
    31:             cand_score = model(cand).gather(1, labels.view(-1, 1)).squeeze(1)
    32:             improve = cand_score < best
    33: 
    34:             if improve.any():
    35:                 mask = improve.view(-1, 1, 1, 1)
    36:                 adv_images = torch.where(mask, cand, adv_images)
    37:                 best = torch.where(improve, cand_score, best)
    38: 
    39:     delta = torch.clamp(adv_images - images, min=-eps, max=eps)
    40:     return torch.clamp(images + delta, 0.0, 1.0).detach()
    41: 
    42: # =====================================================================
    43: # END EDITABLE REGION
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
