# Poison-Robust Learning under Label-Flip Poisoning

## Research Question
How can we design a stronger loss function or sample-weighting rule that improves robustness to poisoned training labels without changing the model, optimizer, or data pipeline?

## Background
A fraction of poisoned (label-flipped) training labels can disproportionately distort model decision boundaries. Robust learning methods typically modify the objective to downweight suspicious samples or reduce memorization of corrupted targets. Representative approaches include the bootstrapping target (Reed et al., ICLR Workshop 2015, arXiv:1412.6596), Generalized Cross Entropy (Zhang and Sabuncu, NeurIPS 2018, arXiv:1805.07836), and Symmetric Cross Entropy (Wang et al., ICCV 2019, arXiv:1908.06112), each of which introduces a saturation or interpolation mechanism that limits the gradient impact of confidently wrong labels.

This task uses research-scale models (ResNet-20, VGG-16-BN, MobileNetV2) trained on full datasets with standard SGD + CosineAnnealing for 100 epochs.

## Task
Implement a better poison-robust objective in `pytorch-vision/bench/poison/custom_robust_loss.py`. The fixed harness injects random label-flip corruption (`(original + 1) % num_classes`) into the training set, trains with your loss, and evaluates on a clean test set.

Your method should improve clean test accuracy under poisoning while reducing how much the model memorizes poisoned labels. The approach must be modular and transferable across architectures and datasets.

## Editable Interface
You must implement:

```python
class RobustLoss:
    def compute_loss(self, logits, labels, epoch):
        ...
```

- `logits`: current minibatch model outputs.
- `labels`: possibly poisoned labels (label-flip: `(original + 1) % num_classes`).
- `epoch`: current training epoch (0-indexed).
- Return value: scalar loss tensor.

The corruption process, model architectures, optimizer, and training schedule are fixed.

## Evaluation
Benchmarks:

- `resnet20-cifar10-labelflip`: ResNet-20 on CIFAR-10, 10% label-flip poison.
- `vgg16bn-cifar100-labelflip`: VGG-16-BN on CIFAR-100, 10% label-flip poison.
- `mobilenetv2-fmnist-labelflip`: MobileNetV2 on FashionMNIST, 15% label-flip poison.

Reported metrics:
- `test_acc`: accuracy on clean test set.
- `poison_fit`: fraction of poisoned samples where model predicts the poisoned (wrong) label.
- `robust_score = (test_acc + (1 - poison_fit)) / 2`.

Primary metric: `robust_score` (higher is better).

## Baselines
The baselines below run inside the same harness via edit ops; defaults follow the corresponding papers:

- `cross_entropy`: standard ERM on poisoned labels.
- `generalized_ce`: Generalized Cross Entropy (Zhang and Sabuncu, NeurIPS 2018, arXiv:1805.07836) with default `q = 0.7`.
- `symmetric_ce`: Symmetric Cross Entropy (Wang et al., ICCV 2019, arXiv:1908.06112), CE plus reverse-CE; CIFAR-10 defaults `alpha = 0.1`, `beta = 1.0`. Reference code: https://github.com/YisenWang/symmetric_cross_entropy_for_noisy_labels.
- `bootstrap`: bootstrapping target interpolation with model predictions (Reed et al., ICLR Workshop 2015, arXiv:1412.6596).
