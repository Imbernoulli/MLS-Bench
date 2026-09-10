# CV Global Pooling / Feature Aggregation Design

## Research Question
Design a global pooling / feature aggregation module for image classification that improves test accuracy across different CNN architectures and datasets, while preserving the surrounding backbone and classifier interface.

## Background
Global pooling is the final spatial aggregation step in modern image-classification CNNs, reducing feature maps from `[B, C, H, W]` to `[B, C]` before the classifier head. The standard choice is Global Average Pooling (GAP), which computes the spatial mean per channel — simple and stable, but treats every spatial location identically and discards the distribution of activations. Alternatives include:

- **Global Max Pooling (GMP)**: selects the strongest activation per channel; captures peak features but ignores other spatial information.
- **Generalized Mean (GeM) Pooling** (Radenović, Tolias & Chum, "Fine-tuning CNN Image Retrieval with No Human Annotation", arXiv:1711.02512, TPAMI 2018): a learnable power-mean `f_p(x) = (mean(x^p))^(1/p)` that interpolates between average pooling (p=1) and max pooling (p→∞). The paper uses an initial value of `p=3.0`.
- **Average + Max**: element-wise sum (or concatenation reduced to `C`) of GAP and GMP, capturing both mean-field and peak statistics.
- Attention- or distribution-based aggregations that learn spatial weights or higher-order statistics.

There is room to design pooling rules that better capture spatial statistics of feature maps, adapt to different architectures, or learn task-specific aggregation patterns.

## What You Can Modify
The `CustomPool` class inside `pytorch-vision/custom_pool.py`. The forward signature takes a `[B, C, H, W]` tensor and must return a `[B, C]` tensor.

You may modify the aggregation function (mean, max, learned weights, attention, higher-order statistics), introduce learnable parameters, choose how spatial information is summarized (single-point, multi-scale, distribution-based), and apply channel-wise or spatial-wise weighting.

Constraints:
- Input shape: `[B, C, H, W]`. `C` varies by architecture (`64` for ResNet-56, `512` for VGG-16-BN, `1280` for MobileNetV2 at this resolution).
- Output shape: `[B, C]`. The output channel dimension must equal the input channel dimension exactly.
- Must work with variable spatial sizes (e.g. `8×8` for ResNet on CIFAR, `1×1` after VGG max-pools / MobileNetV2 stem).
- No access to training data or labels inside the pooling layer.

## Fixed Pipeline
- Optimizer: SGD with `lr=0.1`, `momentum=0.9`, `weight_decay=5e-4`.
- Schedule: cosine annealing over `200` epochs.
- Data augmentation: `RandomCrop(32, pad=4)` + `RandomHorizontalFlip`.
- Evaluation settings: ResNet-56 on CIFAR-100, VGG-16-BN on CIFAR-100, and MobileNetV2 on FashionMNIST.

## Baselines
- **global_max** — channel-wise max over the spatial axes (no extra parameters).
- **gem** — Radenović et al., arXiv:1711.02512; default learnable `p` initialized to `3.0`, with stability epsilon `1e-6`.
- **avg_max** — sum of GAP and GMP outputs (no learnable parameters).

## Metric
Best test accuracy (%, higher is better) achieved during training. The pooling module must accept convolutional feature maps and return the expected channel vector, must handle variable spatial sizes, and must not change datasets, classifier targets, optimizer behavior, or test-time evaluation.
