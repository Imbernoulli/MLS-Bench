# DL Residual Connection Block Design

## Research Question
Design a residual / skip-connection block for CIFAR-style ResNets that improves test accuracy across different network depths and datasets, while keeping the broader training recipe, initialization, data pipeline, optimizer, and classifier objective fixed.

## Background
Residual connections (He et al., "Deep Residual Learning for Image Recognition", arXiv:1512.03385) enabled training of very deep networks by providing identity shortcut paths. The basic residual block adds the input to the output of two stacked 3×3 convolutions. Several improvements have been proposed:

- **Pre-activation ResBlock** (He et al., "Identity Mappings in Deep Residual Networks", ECCV 2016, arXiv:1603.05027): BN-ReLU-Conv ordering, enabling cleaner gradient flow through identity shortcuts.
- **ReZero / gated residual** (Bachlechner et al., "ReZero is All You Need: Fast Convergence at Large Depth", arXiv:2003.04887): a single learnable scalar gate, initialized to `0`, multiplies the residual branch before addition; the network gradually learns the optimal residual contribution per block.
- **Stochastic Depth** (Huang et al., ECCV 2016, arXiv:1603.09382): randomly drops entire residual blocks during training with a linearly decaying survival probability `p_l = 1 − (l/L)(1 − p_L)` and final-block survival `p_L=0.5`; acts as an implicit ensemble regularizer especially effective for very deep networks.
- **ResNeXt** (Xie et al., CVPR 2017, arXiv:1611.05431): grouped convolutions for multi-branch aggregation, with cardinality as a third capacity axis.
- **Res2Net** (Gao et al., TPAMI 2019, arXiv:1904.01169): hierarchical residual-like connections within a single block for multi-scale feature extraction.
- **SE block** (Hu, Shen & Sun, CVPR 2018): channel attention applied inside the residual branch.

There is room for novel block designs that better balance gradient flow, feature reuse, and regularization, particularly across varying network depths.

## What You Can Modify
The `CustomBlock` class inside `pytorch-vision/custom_residual.py`. It is the residual block used by the ResNet backbone.

Constraints (the backbone relies on these):
- Constructor: `CustomBlock(in_planes, planes, stride)`.
- Class attribute `expansion` (`1` for basic, `4` for bottleneck, etc.).
- `forward(x)` returns a tensor with `planes * expansion` channels.
- The shortcut must handle dimension mismatches when `stride != 1` or when the input/output channel count differs.

You may modify the internal convolution structure (number, kernel sizes, grouping), activation/normalization placement and type, the shortcut/skip design, attention mechanisms (channel or spatial), the `expansion` attribute, and any additional modules within the block.

## Fixed Pipeline
- Optimizer: SGD with `lr=0.1`, `momentum=0.9`, `weight_decay=5e-4`.
- Schedule: cosine annealing over `200` epochs.
- Data augmentation: `RandomCrop(32, pad=4)` + `RandomHorizontalFlip`.
- Evaluation settings: ResNet-20 (`[3,3,3]`) on CIFAR-10, ResNet-56 (`[9,9,9]`) on CIFAR-100, ResNet-110 (`[18,18,18]`) on CIFAR-100.

## Baselines
- **pre_activation** — He et al., arXiv:1603.05027; BN-ReLU-Conv ordering inside the block.
- **gated_residual** — ReZero-style learnable scalar gate per block, initialized to `0` (Bachlechner et al., arXiv:2003.04887).
- **stochastic_depth** — Huang et al., arXiv:1603.09382; linearly decaying per-block survival probability with `p_L=0.5`.

## Metric
Best test accuracy (%, higher is better) achieved during training. The block must satisfy the interface above and must not change dataset construction, optimization, global pooling, classifier heads, or the outer training loop.
