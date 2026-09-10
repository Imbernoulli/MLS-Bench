# CV Data Augmentation Strategy Design

## Research Question
Design a training-time data augmentation strategy for image classification that improves test accuracy across different architectures and datasets, while keeping the model architectures, optimizer, test transform, and training loop fixed.

## Background
Data augmentation is a primary regularization tool for training deep networks on limited image data. By applying label-preserving transformations to training images, augmentation increases the effective dataset diversity and shapes the inductive bias of the model. Representative methods include:

- **Standard CIFAR augmentation**: `RandomCrop(32, padding=4)` + `RandomHorizontalFlip` — a minimal geometric baseline.
- **Cutout** (DeVries & Taylor, arXiv:1708.04552): randomly masks square regions of the input, forcing the network to use broader spatial context.
- **RandAugment** (Cubuk et al., CVPR Workshops 2020 / NeurIPS 2020, arXiv:1909.13719): applies `N` randomly selected operations at uniform magnitude `M`, removing the expensive search of AutoAugment-style methods.
- **TrivialAugment** (Müller & Hutter, ICCV 2021, arXiv:2103.10158): applies a single random operation with a random magnitude per image, with no tunable hyperparameters.
- **AugMix** (Hendrycks et al., ICLR 2020, arXiv:1912.02781): mixes multiple augmentation chains for robustness and uncertainty calibration.
- **Random Erasing** (Zhong et al., 2017): an erasing variant closely related to Cutout, often used jointly with other augmentations.

These methods make different choices about geometric, photometric, and masking transforms, and they may behave differently across datasets and model families.

## What You Can Modify
The `build_train_transform(config)` function inside `pytorch-vision/custom_augment.py`. The function receives a `config` dict and must return a `torchvision.transforms.Compose` pipeline.

`config` provides:
- `img_size` (int, `32`)
- `mean` (tuple of channel means)
- `std` (tuple of channel standard deviations)
- `dataset` (str, e.g. `'cifar10'` or `'cifar100'`)

You may use any combination of geometric transforms (crop, flip, rotation, affine, perspective), photometric transforms (color jitter, equalize, posterize, solarize), erasing/masking strategies (cutout, random erasing), automated augmentation policies (AutoAugment, RandAugment, TrivialAugment, AugMix), and custom transform classes defined inside the function. Dataset-specific behavior is allowed.

**Required**: the returned pipeline must include `transforms.ToTensor()` and `transforms.Normalize(config['mean'], config['std'])` so that the produced tensors are normalized as expected by the downstream models. The test-time transform is fixed and is not part of the design space.

## Fixed Pipeline
- Optimizer: SGD with `lr=0.1`, `momentum=0.9`, `weight_decay=5e-4`.
- Schedule: cosine annealing over `200` epochs.
- Weight initialization: standard Kaiming normal.
- Evaluation settings: ResNet-20 on CIFAR-10, ResNet-56 on CIFAR-100, MobileNetV2 on FashionMNIST.

## Baselines
- **cutout** — DeVries & Taylor, arXiv:1708.04552; default 16×16 patch on CIFAR-style 32×32 inputs as in the paper.
- **randaugment** — Cubuk et al., arXiv:1909.13719; default `N=2`, `M=14` (paper-reported defaults for ResNet-style models on CIFAR).
- **trivialaugment** — Müller & Hutter, arXiv:2103.10158; parameter-free, single random op per image with random magnitude.

## Metric
Best test accuracy (%, higher is better) achieved during training. The transform must produce normalized tensors compatible with the existing loaders and models, and must not use validation/test labels, change the dataset split, or alter the model and optimization code.
