# DL Activation Function Design

## Research Question
Design an activation function for deep convolutional neural networks that improves test accuracy across different architectures (ResNet, VGG) and datasets (CIFAR-10, CIFAR-100, FashionMNIST), while keeping the model definitions, optimizer, initialization, and data pipeline fixed.

## Background
Activation functions introduce nonlinearity into neural networks and critically affect training dynamics, gradient flow, sparsity, and generalization. Classic and modern choices include:

- **ReLU** (Nair & Hinton, 2010): `max(0, x)` — simple, sparse, but zero gradient for negative inputs ("dying ReLU").
- **GELU** (Hendrycks & Gimpel, "Gaussian Error Linear Units (GELUs)", arXiv:1606.08415): `x * Phi(x)` where `Phi` is the standard Gaussian CDF; smooth weighting by Gaussian probability mass.
- **Swish / SiLU** (Ramachandran, Zoph & Le, "Searching for Activation Functions", arXiv:1710.05941; SiL form due to Elfwing et al., 2017): `x * sigmoid(beta * x)`; self-gated, smooth, non-monotonic. The PyTorch `nn.SiLU` corresponds to `beta = 1`.
- **Mish** (Misra, "Mish: A Self Regularized Non-Monotonic Activation Function", BMVC 2020, arXiv:1908.08681): `x * tanh(softplus(x))`; self-regularized, smooth, non-monotonic.
- **Squared ReLU**, **StarReLU**, and other variants explore polynomial gates and learnable/affine extensions.

These functions differ in smoothness, gating behavior, and negative-domain treatment, and may interact differently with modern network components such as residual connections and batch normalization.

## What You Can Modify
The `CustomActivation` class inside `pytorch-vision/custom_activation.py`. It is an `nn.Module` used as a drop-in replacement for ReLU throughout the network.

You may modify the `forward` computation (any element-wise or channel-wise operation), register learnable parameters in `__init__`, choose any shape of activation curve (monotonic / non-monotonic / bounded), and decide negative-domain behavior (zero, linear, bounded, learnable). Tensor shape must be preserved.

The activation is used in:
- ResNet: BasicBlock (twice per block) and the initial conv.
- VGG: after every Conv-BN pair and inside the classifier head.
- MobileNetV2: replaces the ReLU6 baseline used in inverted residuals.

## Fixed Pipeline
- Optimizer: SGD with `lr=0.1`, `momentum=0.9`, `weight_decay=5e-4`.
- Schedule: cosine annealing over `200` epochs.
- Data augmentation: `RandomCrop(32, pad=4)` + `RandomHorizontalFlip`.
- Weight initialization: standard Kaiming normal (fixed).
- Evaluation settings: ResNet-20 on CIFAR-10, VGG-16-BN on CIFAR-100, MobileNetV2 on FashionMNIST.

## Baselines
- **gelu** — Hendrycks & Gimpel, arXiv:1606.08415; `nn.GELU` (no learnable parameters).
- **silu** — Ramachandran et al. / Elfwing et al., arXiv:1710.05941; `nn.SiLU`, equivalent to Swish with `beta=1` (no learnable parameters).
- **mish** — Misra, arXiv:1908.08681; `x * tanh(softplus(x))` (no learnable parameters).

## Metric
Best test accuracy (%, higher is better) achieved during training. The activation must be differentiable, shape-preserving, and must not change normalization layers, residual blocks, classifier heads, datasets, or the training loop.
