# CV Multi-Task Loss Combination Strategy Design

## Research Question
Design a multi-task loss combination strategy for jointly training fine-grained (100-class) and coarse (20-superclass) classification on CIFAR-100, with the primary objective of maximizing fine-class test accuracy.

## Background
CIFAR-100 contains 100 fine classes organized into 20 coarse superclasses. Training a model with two classification heads (fine + coarse) provides a natural multi-task learning setup where the coarse task acts as an auxiliary signal. The key challenge is how to combine the two losses so the auxiliary signal helps rather than hurts the primary objective. Representative approaches include:

- **Equal weighting**: simply sum the per-task losses (the trivial baseline).
- **Uncertainty weighting** (Kendall, Gal & Cipolla, CVPR 2018, arXiv:1705.07115): learn a per-task log-variance `s_i` and combine losses as `sum_i (exp(-s_i) * L_i + s_i)`.
- **Dynamic Weight Average (DWA)** (Liu, Johns & Davison, "End-to-End Multi-Task Learning with Attention", CVPR 2019, arXiv:1803.10704): weight each task by the relative rate of change of its loss across recent epochs, with a temperature parameter (`T=2.0` is the value used in the paper).
- **PCGrad** (Yu et al., "Gradient Surgery for Multi-Task Learning", NeurIPS 2020, arXiv:2001.06782): when two task gradients have negative cosine similarity, project each onto the normal plane of the other to reduce gradient interference; otherwise leave them unchanged.
- **Random Loss Weighting**: simple stochastic weighting baseline used as a sanity check in some MTL studies.

The coarse labels encode semantic hierarchy, and balancing this auxiliary signal against the fine-class objective interacts non-trivially with architecture, training stage, and gradient geometry.

## What You Can Modify
The `MultiTaskLoss` class inside `pytorch-vision/custom_mtl.py`. The class receives the individual task losses and returns a single scalar loss.

The `forward` method receives:
- `fine_loss` (scalar tensor): cross-entropy for the 100-class fine head.
- `coarse_loss` (scalar tensor): cross-entropy for the 20-class coarse head.
- `epoch` (int): current epoch (0-indexed).
- `total_epochs` (int): total number of training epochs.

You may modify `__init__` to add learnable parameters (log-variances, weights, etc.), implement any combination strategy in `forward`, use `epoch` / `total_epochs` for curriculum or scheduling, and maintain auxiliary state such as loss-history buffers. The `MultiTaskLoss` parameters are included in the optimizer, so any registered learnable tensors will be trained.

## Fixed Pipeline
- Optimizer: SGD with `lr=0.1`, `momentum=0.9`, `weight_decay=5e-4`.
- Schedule: cosine annealing over `200` epochs.
- Data augmentation: `RandomCrop(32, pad=4)` + `RandomHorizontalFlip`.
- Two-head model: shared backbone with separate fine (100-way) and coarse (20-way) classifiers.
- Evaluation settings: ResNet-20, ResNet-56 (CIFAR-100, fine+coarse heads), and VGG-16-BN on CIFAR-100 with the same fine+coarse setup.

## Baselines
- **uncertainty** — Kendall et al., arXiv:1705.07115; learns one log-variance per task initialized to `0`.
- **dwa** — Liu et al., arXiv:1803.10704; default temperature `T=2.0`.
- **pcgrad** — Yu et al., arXiv:2001.06782; project conflicting fine/coarse gradients on each parameter group.

## Metric
Best fine-class test accuracy (%, higher is better) achieved during training. The combination module must remain differentiable and must not change labels, heads, datasets, model backbones, or the outer training loop.
