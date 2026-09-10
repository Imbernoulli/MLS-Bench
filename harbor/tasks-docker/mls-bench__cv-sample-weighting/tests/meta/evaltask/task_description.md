# CV Sample Reweighting Strategy Design

## Research Question
Design a class-weighting strategy for class-imbalanced image classification that improves balanced test accuracy on long-tail distributed datasets, across different architectures and imbalance ratios, while keeping the dataset construction, sampler, model, optimizer, and evaluation metric fixed.

## Background
Real-world datasets often follow long-tail class distributions: a few "head" classes dominate while many "tail" classes have very few samples. Uniform cross-entropy biases the classifier toward frequent classes, degrading performance on rare ones. Class reweighting assigns per-class weights to the cross-entropy loss to counteract this imbalance. Representative formulations include:

- **Inverse frequency**: `w[c] = N / (C * n[c])`, directly compensating for class size.
- **Square-root inverse**: `w[c] ∝ 1 / sqrt(n[c])`, a smoother variant that under-weights extreme rare-class amplification.
- **Effective Number of Samples** (Cui et al., CVPR 2019, arXiv:1901.05555): models data overlap with `E_n = (1 - β^n) / (1 - β)` and uses `w[c] ∝ 1 / E_{n[c]}`; the paper reports `β ∈ {0.9, 0.99, 0.999, 0.9999}` with `β=0.9999` typical for long-tail CIFAR.
- **Balanced Softmax-style weighting** (Ren et al., "Balanced Meta-Softmax for Long-Tailed Visual Recognition", NeurIPS 2020, arXiv:2007.10740): rebalances the softmax via a prior derived from class frequencies; equivalent in our setting to a particular weighting form on the loss.
- **LDAM** (Cao et al., NeurIPS 2019, arXiv:1906.07413): a related label-distribution-aware margin formulation, often combined with deferred reweighting.

These methods define different mappings from class frequency to loss weight, and may behave differently across datasets and imbalance regimes.

## What You Can Modify
The `compute_class_weights(class_counts, num_classes, config)` function inside `pytorch-vision/custom_weighting.py`. The function receives per-class sample counts and must return a 1-D tensor of length `num_classes` suitable for `nn.CrossEntropyLoss(weight=...)`.

`config` provides:
- `imbalance_ratio` (float)
- `dataset` (str)
- `arch` (str)
- `total_samples` (int)

You may modify the functional form mapping class counts to weights (inverse, power-law, logarithmic, piecewise, effective-number, etc.), use any field from `config`, choose any normalization strategy (sum to `C`, sum to `1`, unnormalized), and combine multiple ideas. The computation must be pure: no access to training data, model parameters, or test labels.

## Fixed Pipeline
- Optimizer: SGD with `lr=0.1`, `momentum=0.9`, `weight_decay=5e-4`.
- Schedule: cosine annealing over `200` epochs.
- Data augmentation: `RandomCrop(32, pad=4)` + `RandomHorizontalFlip`.
- Evaluation is on the balanced test set; training is on the long-tail train split.
- Evaluation settings: ResNet-32 on CIFAR-10-LT (imbalance ratio 100), ResNet-32 on CIFAR-100-LT (imbalance ratio 100), and VGG-16-BN on CIFAR-100-LT (imbalance ratio 50).

## Baselines
- **inverse_freq** — `w[c] = total_samples / (num_classes * n[c])`.
- **effective_number** — Cui et al., arXiv:1901.05555; default `β=0.9999` (paper-recommended for long-tail CIFAR-100), with weights normalized so they sum to `num_classes`.
- **balanced_softmax** — weighting form motivated by Ren et al., arXiv:2007.10740, derived from the empirical class prior.

## Metric
Best test accuracy (%, higher is better) on the balanced test set. The weighting rule must produce numerically stable class weights compatible with cross-entropy and must not change the dataset construction, sampler, model architecture, optimizer, or evaluation metric.
