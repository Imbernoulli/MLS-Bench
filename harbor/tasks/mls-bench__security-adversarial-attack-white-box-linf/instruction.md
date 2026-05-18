# MLS-Bench: security-adversarial-attack-white-box-linf

# White-Box Evasion Attack under Linf Constraint

## Research Question
Can you design a stronger white-box `L_inf` evasion attack that increases attack success rate (ASR) under a small `eps` budget, where weak attacks already saturate near 100% on undefended models but strong baselines (PGD, AutoAttack) leave headroom on some architectures?

## Background
White-box evasion attacks assume the attacker has full access to the model, including its parameters and gradients. The classical first-order attack is FGSM (Goodfellow et al., 2015, arXiv:1412.6572), a one-step sign-of-gradient attack. Iterative variants such as PGD (Madry et al., 2018, arXiv:1706.06083) and momentum iterative FGSM, MI-FGSM (Dong et al., CVPR 2018, arXiv:1710.06081), refine the perturbation through multiple gradient steps. AutoAttack (Croce and Hein, ICML 2020, arXiv:2003.01690) is a parameter-free ensemble of two adaptive PGD variants together with FAB and Square Attack and is widely used as a strong reference attack.

## Objective
Implement a stronger white-box `L_inf` attack in `bench/custom_attack.py`. The method should maximize ASR under a strict perturbation budget:

- Threat model: white-box (full model access, including gradients).
- Norm constraint: `||x_adv - x||_inf <= eps`.
- Budget: `eps = 2/255`. RobustBench uses `8/255` for *defended* models, which saturates ASR to ~1.0 on undefended models and leaves no headroom for agents; the `2/255` regime is used here to differentiate attack quality on undefended classifiers.

## Editable Interface
You must implement:

`run_attack(model, images, labels, eps, device, n_classes) -> adv_images`

Inputs:
- `images`: tensor of shape `(N, C, H, W)`, values in `[0, 1]`.
- `labels`: tensor of shape `(N,)`.
- `n_classes`: 10 for CIFAR-10, 100 for CIFAR-100.

Output:
- `adv_images`: same shape as `images`, also in `[0, 1]`.

## Evaluation Protocol
Each evaluation script:
1. Loads one pretrained model.
2. Collects up to 1000 samples that are initially classified correctly.
3. Runs your `run_attack`.
4. Checks `L_inf` validity and `[0, 1]` range.
5. Reports `clean_acc`, `robust_acc`, and `asr = 1 - robust_acc`.

Important:
- ASR denominator is the number of initially correct samples.
- Invalid adversarial outputs (shape mismatch, non-finite values, or violated norm) are treated as failure.

## Evaluation Scenarios
Each scenario is a (model, dataset) pair drawn from {ResNet20, VGG11-BN, MobileNetV2} x {CIFAR-10, CIFAR-100}, using publicly available pretrained checkpoints.

## Baselines
The baselines below run inside the same harness via edit ops; reference implementations are in `torchattacks`:

- `fgsm`: Fast Gradient Sign Method (Goodfellow et al., 2015, arXiv:1412.6572). One-step sign-of-gradient attack.
- `pgd`: PGD (Madry et al., 2018, arXiv:1706.06083). Iterative projected gradient descent on the cross-entropy loss with random start, 40 inner steps and step size `eps/4`.
- `mifgsm`: MI-FGSM (Dong et al., CVPR 2018, arXiv:1710.06081). Iterative FGSM with momentum on the gradient direction.
- `autoattack`: AutoAttack (Croce and Hein, ICML 2020, arXiv:2003.01690). `torchattacks.AutoAttack(version="standard")`, the parameter-free ensemble of APGD-CE, APGD-DLR, FAB and Square Attack.

## Note on per-architecture natural robustness
At `eps=2/255`, ASR differs substantially across architectures on undefended models. This is an architectural property, not an evaluation bug: VGG-style wider-but-shallower activations at low-resolution feature maps absorb small `L_inf` perturbations more robustly than bottlenecked ResNet or depthwise-separable MobileNetV2, so PGD and AutoAttack saturate on the latter but leave meaningful headroom on the former.


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
    11:     eps: float,
    12:     device: torch.device,
    13:     n_classes: int,
    14: ) -> torch.Tensor:
    15:     """
    16:     White-box L_inf adversarial attack.
    17:     images: (N, C, H, W) in [0, 1] on device. labels: (N,) on device.
    18:     n_classes: 10 for CIFAR-10, 100 for CIFAR-100.
    19:     Returns adv_images satisfying (adv_images - images).abs().max() <= eps.
    20:     """
    21:     _ = (model, labels, eps, device, n_classes)
    22:     return images.clone()
    23: 
    24: # =====================================================================
    25: # END EDITABLE REGION
    26: # =====================================================================
```




## How You Will Be Evaluated

After you finish, evaluation runs a fixed set of scripts and aggregates the
metrics they emit. These scripts are **not** in your workspace — you cannot
read or modify them. The labels below indicate what each evaluation tests:

- **ResNet20-C10** — wall-clock budget `0:20:00`, compute share `0.5`
- **VGG11BN-C10** — wall-clock budget `0:20:00`, compute share `0.5`
- **ResNet20-C100** — wall-clock budget `0:20:00`, compute share `0.5`
- **VGG11BN-C100** — wall-clock budget `0:20:00`, compute share `0.5`
- **MobileNetV2-C100** — wall-clock budget `0:20:00`, compute share `0.5`


Scoring uses the same `combined_score` aggregation as the MLS-Bench
leaderboard. Multiple seeds are averaged.



## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `fgsm` baseline — editable region  [READ-ONLY — reference implementation]

In `torchattacks/bench/custom_attack.py`:

```python
Lines 3–33:
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
    12:     device: torch.device,
    13:     n_classes: int,
    14: ) -> torch.Tensor:
    15:     import torch.nn.functional as F
    16: 
    17:     _ = (device, n_classes)
    18:     model.eval()
    19:     x = images.detach().clone().requires_grad_(True)
    20:     logits = model(x)
    21:     loss = F.cross_entropy(logits, labels)
    22:     grad = torch.autograd.grad(loss, x)[0]
    23: 
    24:     with torch.no_grad():
    25:         x_adv = x + eps * grad.sign()
    26:         delta = torch.clamp(x_adv - images, min=-eps, max=eps)
    27:         x_adv = torch.clamp(images + delta, 0.0, 1.0)
    28: 
    29:     return x_adv.detach()
    30: 
    31: # =====================================================================
    32: # END EDITABLE REGION
    33: # =====================================================================
```

### `pgd` baseline — editable region  [READ-ONLY — reference implementation]

In `torchattacks/bench/custom_attack.py`:

```python
Lines 3–43:
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
    12:     device: torch.device,
    13:     n_classes: int,
    14: ) -> torch.Tensor:
    15:     import torch.nn.functional as F
    16: 
    17:     _ = (device, n_classes)
    18:     model.eval()
    19:     steps = 40
    20:     alpha = eps / 4.0
    21: 
    22:     x = images.detach()
    23:     x_adv = x + torch.empty_like(x).uniform_(-eps, eps)
    24:     x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
    25: 
    26:     for _ in range(steps):
    27:         x_adv.requires_grad_(True)
    28:         logits = model(x_adv)
    29:         loss = F.cross_entropy(logits, labels)
    30:         grad = torch.autograd.grad(loss, x_adv)[0]
    31: 
    32:         with torch.no_grad():
    33:             x_adv = x_adv + alpha * grad.sign()
    34:             delta = torch.clamp(x_adv - x, min=-eps, max=eps)
    35:             x_adv = torch.clamp(x + delta, 0.0, 1.0)
    36: 
    37:         x_adv = x_adv.detach()
    38: 
    39:     return x_adv
    40: 
    41: # =====================================================================
    42: # END EDITABLE REGION
    43: # =====================================================================
```

### `mifgsm` baseline — editable region  [READ-ONLY — reference implementation]

In `torchattacks/bench/custom_attack.py`:

```python
Lines 3–47:
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
    12:     device: torch.device,
    13:     n_classes: int,
    14: ) -> torch.Tensor:
    15:     import torch.nn.functional as F
    16: 
    17:     _ = (device, n_classes)
    18:     model.eval()
    19:     steps = 40
    20:     alpha = eps / 10.0
    21:     decay = 1.0
    22: 
    23:     x = images.detach()
    24:     x_adv = x + torch.empty_like(x).uniform_(-eps, eps)
    25:     x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
    26:     momentum = torch.zeros_like(x)
    27: 
    28:     for _ in range(steps):
    29:         x_adv.requires_grad_(True)
    30:         logits = model(x_adv)
    31:         loss = F.cross_entropy(logits, labels)
    32:         grad = torch.autograd.grad(loss, x_adv)[0]
    33:         grad = grad / (grad.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-12)
    34:         momentum = decay * momentum + grad
    35: 
    36:         with torch.no_grad():
    37:             x_adv = x_adv + alpha * momentum.sign()
    38:             delta = torch.clamp(x_adv - x, min=-eps, max=eps)
    39:             x_adv = torch.clamp(x + delta, 0.0, 1.0)
    40: 
    41:         x_adv = x_adv.detach()
    42: 
    43:     return x_adv
    44: 
    45: # =====================================================================
    46: # END EDITABLE REGION
    47: # =====================================================================
```

### `autoattack` baseline — editable region  [READ-ONLY — reference implementation]

In `torchattacks/bench/custom_attack.py`:

```python
Lines 3–33:
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
    12:     device: torch.device,
    13:     n_classes: int,
    14: ) -> torch.Tensor:
    15:     import os
    16:     import torchattacks
    17: 
    18:     _ = device
    19:     model.eval()
    20:     attack = torchattacks.AutoAttack(
    21:         model,
    22:         norm="Linf",
    23:         eps=eps,
    24:         version="standard",
    25:         n_classes=n_classes,
    26:         seed=int(os.environ.get("SEED", "42")),
    27:         verbose=False,
    28:     )
    29:     return attack(images, labels)
    30: 
    31: # =====================================================================
    32: # END EDITABLE REGION
    33: # =====================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
