# DL Learning Rate Schedule Design

## Research Question
Design a learning-rate schedule for training deep convolutional image classifiers that improves convergence speed and final test accuracy across different architectures and datasets, while keeping the optimizer type, training loop, and all other hyperparameters fixed.

## Background
Learning-rate scheduling is critical for training deep networks effectively: a fixed learning rate is often too high (unstable) or too low (slow). Representative schedules include:

- **Step decay** (He et al., "Deep Residual Learning for Image Recognition", arXiv:1512.03385): divide the learning rate by 10 at fixed milestones (e.g. 50% and 75% of training).
- **Cosine annealing** (Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts", ICLR 2017, arXiv:1608.03983): smooth decay following a cosine curve from `base_lr` down to a small final value (often `0`).
- **Warmup + cosine** (Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour", arXiv:1706.02677): linear warmup over a small number of epochs followed by cosine (or step) decay; stabilizes large-batch / high learning rates.
- **One-Cycle / Super-convergence** (Smith & Topin, "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates", arXiv:1708.07120): a single triangular ramp up then ramp down, often combined with momentum cycling.
- **Polynomial decay**: `lr(t) = base_lr * (1 - t/T)^p`, common in segmentation literature.

These schedules are usually designed without considering architecture-specific properties (depth, residual structure, BatchNorm) or dataset characteristics; there is room for schedules that adapt to context.

## What You Can Modify
The `get_lr(epoch, total_epochs, base_lr, config)` function inside `pytorch-vision/custom_schedule.py`. The function is called once per epoch and must return the learning rate (a float) used by SGD for that epoch.

`config` provides:
- `arch` (str: e.g. `'resnet20'`, `'resnet56'`, `'mobilenetv2'`)
- `dataset` (str: e.g. `'cifar10'`, `'cifar100'`, `'fmnist'`)

You may freely shape the LR curve (cosine, polynomial, exponential, linear, piecewise), include warmup of arbitrary length and shape, set any minimum/final LR, condition on `arch` and `dataset`, and use any epoch-dependent logic such as cyclic restarts, sharp transitions, or plateaus.

## Fixed Pipeline
- Optimizer: SGD with `lr=base_lr=0.1`, `momentum=0.9`, `weight_decay=5e-4`. The task setup uses **no** built-in PyTorch scheduler — your `get_lr` directly determines the per-epoch learning rate.
- Training: `200` epochs.
- Data augmentation: `RandomCrop(32, pad=4)` + `RandomHorizontalFlip`.
- Weight initialization: Kaiming normal (fixed, not editable).
- Evaluation settings: ResNet-20 on CIFAR-10, ResNet-56 on CIFAR-100, MobileNetV2 on FashionMNIST.

## Baselines
- **cosine** — Loshchilov & Hutter, arXiv:1608.03983; standard cosine annealing from `base_lr` to `0` over `total_epochs`.
- **warmup_cosine** — Goyal et al., arXiv:1706.02677; linear warmup (commonly 5 epochs) followed by cosine annealing to `0`.
- **one_cycle** — Smith & Topin, arXiv:1708.07120; triangular up-then-down ramp that peaks above `base_lr` and ends below it.

## Metric
Best test accuracy (%, higher is better) achieved during training. The schedule must not modify model code, data augmentation, loss functions, optimizer type, weight decay, or evaluation.
