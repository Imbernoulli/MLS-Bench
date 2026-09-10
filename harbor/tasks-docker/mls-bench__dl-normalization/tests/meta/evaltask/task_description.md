# DL Normalization Layer Design

## Research Question
Design a normalization layer for deep convolutional neural networks that improves training stability and final test accuracy across different architectures and datasets, while keeping the optimizer, data pipeline, and outer training loop fixed.

## Background
Normalization layers are critical in modern deep networks: they control activation scale, mitigate internal covariate shift, and enable stable training at higher learning rates. Representative methods include:

- **BatchNorm** (Ioffe & Szegedy, ICML 2015, arXiv:1502.03167): normalizes across the batch dimension per channel; the de facto standard, but depends on batch statistics and behaves differently at train and test time.
- **GroupNorm** (Wu & He, ECCV 2018, arXiv:1803.08494): divides channels into groups and normalizes within each group; batch-size independent.
- **InstanceNorm** (Ulyanov, Vedaldi & Lempitsky, "Instance Normalization: The Missing Ingredient for Fast Stylization", arXiv:1607.08022): normalizes each channel independently per instance; common in style transfer.
- **LayerNorm** (Ba, Kiros & Hinton, arXiv:1607.06450): normalizes across all channels for each sample; standard in transformers.
- **RMSNorm** (Zhang & Sennrich, NeurIPS 2019, arXiv:1910.07467): normalizes by root-mean-square only (no mean centering); cheaper than LayerNorm.
- **Batch-Instance Norm (BIN)** (Nam & Kim, NeurIPS 2018, arXiv:1805.07925): per-channel learnable mixture of BatchNorm and InstanceNorm.
- **Switchable Normalization (SN)** (Luo et al., "Differentiable Learning-to-Normalize via Switchable Normalization", arXiv:1806.10779): learnable convex combination of BN/LN/IN statistics per layer.
- **EvoNorm** (Liu et al., "Evolving Normalization-Activation Layers", NeurIPS 2020, arXiv:2004.02967): jointly evolves normalization and activation.

Each method has limitations: BatchNorm degrades with small batches, GroupNorm requires choosing the number of groups, InstanceNorm discards inter-channel information, and LayerNorm may not suit spatial feature maps. There is room for designs that combine strengths or use novel normalization statistics.

## What You Can Modify
The `CustomNorm` class inside `pytorch-vision/custom_norm.py`. It must be a drop-in replacement for `nn.BatchNorm2d`:

- Constructor: `CustomNorm(num_features)` where `num_features` is the channel count `C`.
- Input shape: `[B, C, H, W]`. Output shape: `[B, C, H, W]`.
- Train and eval behavior must be numerically stable.

You may modify normalization statistics (mean/variance over batch, channel, spatial, or any combination), learnable affine parameters (scale and shift), grouping strategies, mixtures of normalization approaches, and adaptive or input-dependent normalization, as long as the interface is preserved.

## Fixed Pipeline
- Optimizer: SGD with `lr=0.1`, `momentum=0.9`, `weight_decay=5e-4`.
- Schedule: cosine annealing over `200` epochs.
- Data augmentation: `RandomCrop(32, pad=4)` + `RandomHorizontalFlip`.
- Evaluation settings: ResNet-56 on CIFAR-100, MobileNetV2 on FashionMNIST, and ResNet-110 on CIFAR-100.

## Baselines
- **group_norm** — Wu & He, arXiv:1803.08494; default `num_groups=32` (paper-recommended), with channel counts smaller than `num_groups` falling back to InstanceNorm-equivalent grouping.
- **batch_instance_norm** — Nam & Kim, arXiv:1805.07925; per-channel learnable gate `rho` initialized to `1.0` (BatchNorm-leaning, matching the paper).
- **switchable_norm** — Luo et al., arXiv:1806.10779; learnable softmax weights over `{BN, LN, IN}` statistics per layer.

## Metric
Best test accuracy (%, higher is better) achieved during training. The normalization module must preserve tensor shape, accept the expected channel count, remain numerically stable in train and eval, and must not change backbones, activations, datasets, loss functions, or optimizer settings.
