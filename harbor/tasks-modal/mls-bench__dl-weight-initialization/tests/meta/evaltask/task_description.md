# DL Weight Initialization Strategy Design

## Research Question
Design a data-independent weight initialization strategy for deep convolutional neural networks that improves convergence speed and final test accuracy across different architectures and datasets, while keeping the data pipeline, optimizer, schedule, loss, and model definitions fixed.

## Background
Weight initialization is fundamental to training deep networks. Poor initialization leads to vanishing or exploding gradients, slow convergence, or worse generalization. Representative methods include:

- **Kaiming / He initialization** (He et al., "Delving Deep into Rectifiers", ICCV 2015, arXiv:1502.01852): for ReLU-style nonlinearities, draws conv weights from `N(0, sqrt(2 / fan_mode))` (typically `fan_in` or `fan_out`).
- **Orthogonal initialization** (Saxe, McClelland & Ganguli, ICLR 2014, arXiv:1312.6120): preserves signal norms via random orthogonal matrices, motivated by the dynamics of deep linear networks.
- **Fixup** (Zhang, Dauphin & Ma, ICLR 2019, arXiv:1901.09321): for residual networks without normalization. Scales the last conv in each residual block by `L^(-1/(2m-2))` where `L` is the number of residual blocks and `m` is the number of conv layers per block (commonly `2`); zero-initializes the last conv per block so residual branches start near identity; adds learnable scalar biases / multipliers around each conv.
- **Zero / near-zero residual init** (subset of Fixup-style ideas): zero-initialize the last weight in each residual branch.
- **LSUV** (Mishkin & Matas, 2015): orthogonal init followed by per-layer rescaling of activation variance to `1` using a small calibration batch.

Each of these addresses one aspect of initialization; there is room for strategies that jointly account for residual structure, BatchNorm's rescaling effect, depth, and the interaction between conv and classifier layers.

## What You Can Modify
The `initialize_weights(model, config)` function inside `pytorch-vision/custom_init.py`. The function receives the fully constructed model and a `config` dict, and must initialize all parameters in place.

`config` provides:
- `arch` (str)
- `num_classes` (int)
- `depth` (int): number of `Conv2d` + `Linear` layers in the model.

You may iterate over `model.named_modules()` or `model.named_parameters()` and design per-layer or depth-dependent strategies, treat residual shortcut projections separately from main-path convs, set `BatchNorm2d` weight/bias differently, and use any data-independent logic. No access to training data and no calibration passes.

## Fixed Pipeline
- Optimizer: SGD with `lr=0.1`, `momentum=0.9`, `weight_decay=5e-4`.
- Schedule: cosine annealing over `200` epochs.
- Data augmentation: `RandomCrop(32, pad=4)` + `RandomHorizontalFlip`.
- Evaluation settings: ResNet-56 on CIFAR-100, VGG-16-BN on CIFAR-100, MobileNetV2 on FashionMNIST.

## Baselines
- **kaiming_normal** — He et al., arXiv:1502.01852; conv weights from `N(0, sqrt(2/fan_out))`, zero biases, BatchNorm `(weight=1, bias=0)`.
- **fixup** — Zhang et al., arXiv:1901.09321; scales the first residual conv by `L^(-1/(2m-2))` with `m=2` and zero-initializes the last conv per residual block.
- **orthogonal** — Saxe et al., arXiv:1312.6120; orthogonal init for conv and linear layers (gain `sqrt(2)` for ReLU), zero biases, BatchNorm `(weight=1, bias=0)`.

## Metric
Best test accuracy (%, higher is better) achieved during training. The initialization must be data-independent and must not run calibration passes, alter the model graph, change optimizer hyperparameters, or modify evaluation behavior.
