# DL Regularization Strategy Design

## Research Question
Design an additional regularization term for deep convolutional image classifiers that improves generalization (test accuracy) across different architectures and datasets, while the main cross-entropy objective, optimizer, and outer training loop remain fixed.

## Background
Beyond standard weight decay (L2 penalty applied through the optimizer), many regularization techniques have been proposed to improve generalization in deep networks:

- **DropBlock** (Ghiasi, Lin & Le, NeurIPS 2018, arXiv:1810.12890): drops contiguous regions of feature maps. Reformulating its core insight as a loss-based penalty yields a spatial co-activation penalty that discourages reliance on contiguous regions without modifying the model graph.
- **Confidence penalty** (Pereyra et al., "Regularizing Neural Networks by Penalizing Confident Output Distributions", arXiv:1701.06548): adds a penalty `−H(p_θ(y|x))` to discourage low-entropy output distributions.
- **Orthogonal regularization** (Brock et al., "Neural Photo Editing with Introspective Adversarial Networks", ICLR 2017, arXiv:1609.07093): encourages weight matrices to be orthogonal via `||W^T W − I||_F^2` (or its soft variants), preserving signal norms.
- **Spectral / Frobenius penalties**: bound the Lipschitz constant or norms of layer weights.

These methods typically apply a fixed penalty throughout training and do not adapt to training dynamics, model architecture, or interactions between different layer types. There is room for regularizers that are more adaptive, architecture-aware, or that combine complementary penalties.

## What You Can Modify
The `compute_regularization(model, inputs, outputs, targets, config)` function inside `pytorch-vision/custom_reg.py`. The function is called every training step and returns a scalar tensor that is added to the cross-entropy loss.

Inputs:
- `model`: the full `nn.Module`. Iterate over `model.named_parameters()` or `model.named_modules()` for weight-based penalties.
- `inputs`: `[B, 3, 32, 32]` input batch (for input-dependent regularization).
- `outputs`: `[B, num_classes]` model logits (for output-based penalties such as confidence/entropy).
- `targets`: `[B]` integer class labels.
- `config`: dict with `num_classes` (int), `epoch` (int, 0-indexed), `total_epochs` (int).

Design directions: weight-based (L1/L2 norms, orthogonality, spectral norms, weight correlation), output-based (entropy, confidence penalty, label-smoothing-style penalties, logit penalties), activation-based (sparsity, diversity via forward hooks), epoch-dependent (warm-up schedules, annealing, curriculum), or architecture-aware (different penalties for conv vs linear, depth-dependent scaling). The returned term must be differentiable.

Note: standard L2 weight decay (`5e-4`) is **already** applied via the optimizer. Your regularization term is *additional*.

## Fixed Pipeline
- Optimizer: SGD with `lr=0.1`, `momentum=0.9`, `weight_decay=5e-4`.
- Schedule: cosine annealing over `200` epochs.
- Data augmentation: `RandomCrop(32, pad=4)` + `RandomHorizontalFlip`.
- Evaluation settings: ResNet-56 on CIFAR-100, VGG-16-BN on CIFAR-100, MobileNetV2 on FashionMNIST.

## Baselines
- **dropblock** — Ghiasi et al., arXiv:1810.12890; loss-based DropBlock-inspired co-activation penalty.
- **confidence_penalty** — Pereyra et al., arXiv:1701.06548; default penalty weight `beta=0.1` (within the `[0.1, 1.0]` range explored in the paper).
- **orthogonal_reg** — Brock et al., arXiv:1609.07093; soft orthogonality penalty `||W^T W − I||_F^2` on conv weights with default coefficient `1e-4`.

## Metric
Best test accuracy (%, higher is better) achieved during training. The regularizer must remain differentiable, computationally reasonable, and must not alter the dataset, architecture, base loss, optimizer, scheduler, or evaluation procedure.
