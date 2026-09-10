# JEPA Self-Supervised Learning: Anti-Collapse Regularization

## Research Question
Design an improved anti-collapse regularization loss for Joint Embedding Predictive Architecture (JEPA) self-supervised image representation learning. Your regularizer should prevent representation collapse (where all inputs map to the same output) while encouraging the model to learn useful, discriminative features.

## Background
JEPA / joint-embedding self-supervised methods (Assran et al., I-JEPA, CVPR 2023, arXiv:2301.08243) optimize an invariance objective that, on its own, admits the trivial solution where the encoder maps every input to a constant. Anti-collapse regularizers solve this in different ways:
- **VICReg** (Bardes, Ponce, LeCun, ICLR 2022, arXiv:2105.04906) combines a per-dimension variance hinge, a covariance off-diagonal penalty, and an MSE invariance term.
- **Barlow Twins** decorrelates the cross-correlation matrix between two views.
- **Whitening / decorrelation** approaches enforce identity covariance directly.

The choice of regularizer determines what representation geometry is preferred and how it transfers to downstream linear probing.

## What You Can Modify
The editable region in `custom_regularizer.py` is the `CustomRegularizer` class plus the `CONFIG_OVERRIDES` dictionary. The class receives two projected embedding tensors from different augmented views of the same images and must return a loss dictionary.

Interface:
- **Input**: `z1: [B, D]` and `z2: [B, D]` — projected embeddings from two augmented views
- **Output**: `dict` with at least a `"loss"` key containing a scalar tensor

You may add any parameters to `__init__`, define helper methods, and use any PyTorch operations. The imports at the top of the file (torch, torch.nn, torch.nn.functional, etc.) are available.

## Evaluation
- **Metric**: `val_acc` — linear probe classification accuracy on CIFAR-10 (higher is better)
- **Benchmarks**: three backbone architectures (ResNet-18, ResNet-34, ResNet-50) test regularizer generalization across model scales
- **Projector**: features_dim → 2048 → 2048 MLP
- **Training**: 100 epochs, batch size 256, LARS optimizer (lr=0.3), warmup cosine schedule
- **Dataset**: CIFAR-10 (50k train / 10k val)
