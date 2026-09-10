# Membership Inference Defense via Training Regularization

## Research Question
How can we design a stronger training-time regularizer that reduces membership inference leakage while preserving standard predictive accuracy?

## Background
Membership inference attacks exploit the gap between train and non-train examples, often through confidence or loss statistics. Many defenses regularize predictions to reduce overconfidence and shrink the train-test generalization gap, but stronger privacy often hurts utility. Common training-time regularizers include label smoothing, the confidence penalty (Pereyra et al., ICLR Workshop 2017, arXiv:1701.06548) which penalizes low-entropy outputs, and RelaxLoss (Chen, Yu, Fritz, ICLR 2022, arXiv:2207.05801), which relaxes the per-example training objective once confidence is already high so that training and held-out distributions become harder to distinguish.

## Task
Implement a better privacy-preserving training objective in `bench/membership/custom_membership_defense.py`. The fixed harness will train a model on a 50/50 train/non-train split of the full dataset, then run a confidence-based membership inference attack on train versus held-out examples.

Your method should improve the privacy-utility tradeoff: lower membership-attack AUC while retaining high test accuracy.

## Editable Interface
You must implement:

```python
class MembershipDefense:
    def compute_loss(self, logits, labels, epoch):
        ...
```

- `logits`: model outputs for the current minibatch.
- `labels`: ground-truth labels.
- `epoch`: current training epoch (0-indexed).
- Return value: scalar loss tensor used by the fixed training loop.

The optimizer (SGD + CosineAnnealing), architecture, data pipeline, and attack implementation are fixed.

## Evaluation
Benchmarks:

- `resnet20-cifar10`: ResNet-20 on CIFAR-10.
- `vgg16bn-cifar100`: VGG-16-BN on CIFAR-100.
- `mobilenetv2-fmnist`: MobileNetV2 on FashionMNIST.

Reported metrics:
- `test_acc`
- `mia_auc`
- `privacy_gap`
- `privacy_score`

Primary metric: `privacy_score = test_acc - max(mia_auc - 0.5, 0)` (higher is better).

## Baselines
The baselines below run inside the same harness via edit ops; defaults follow the corresponding papers:

- `erm`: standard cross-entropy training.
- `label_smoothing`: smoothed targets (smoothing factor `0.1`).
- `confidence_penalty`: cross-entropy plus predictive entropy penalty (Pereyra et al., ICLR Workshop 2017, arXiv:1701.06548) with default penalty weight `0.1`.
- `relaxloss`: margin-aware loss relaxation (Chen, Yu, Fritz, ICLR 2022, arXiv:2207.05801). Reference code: https://github.com/DingfanChen/RelaxLoss.
