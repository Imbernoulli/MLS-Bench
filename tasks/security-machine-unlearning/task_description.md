# Machine Unlearning via Targeted Update Rules

## Research Question
How can we design a stronger unlearning update rule that removes information about a forget set while retaining as much utility as possible on the retained data?

## Background
Machine unlearning methods approximate the effect of retraining without the deleted data. The central tradeoff is clear: aggressive forgetting reduces utility, while conservative updates leave measurable traces of the forgotten examples. Approximate-unlearning approaches range from continued retain-only finetuning, to gradient ascent on the forget loss (NegGrad / "Eternal Sunshine of the Spotless Net": Golatkar, Achille, Soatto, CVPR 2020, arXiv:1911.04933), to incompetent-teacher distillation (Bad-T: Chundawat et al., AAAI 2023, arXiv:2205.08096), and selective student-teacher scrubbing (SCRUB: Kurmanji et al., NeurIPS 2023, arXiv:2302.09880).

The harness pretrains a standard vision model (ResNet-20, VGG-16-BN, or MobileNetV2) on the full training set for 80 epochs using SGD with cosine annealing. After pretraining, a single class is designated as the forget set. The unlearning method then runs for 20 epochs, receiving both retain-set and forget-set minibatches each step, with an Adam optimizer (`lr = 0.001`).

## Task
Implement a better unlearning rule in `pytorch-vision/bench/unlearning/custom_unlearning.py`. The fixed harness trains an initial model, defines a forget split, and then applies your update rule for a fixed number of unlearning steps using retain and forget minibatches.

Your method should lower forget-set memorization while preserving retained-task accuracy.

## Editable Interface
You must implement:

```python
class UnlearningMethod:
    def unlearn_step(self, model, retain_batch, forget_batch, optimizer, step, epoch):
        ...
```

- `retain_batch`: `(images, labels)` tuple from retained data (already on device).
- `forget_batch`: `(images, labels)` tuple from the forget set (already on device).
- `optimizer`: fixed Adam optimizer instance (`lr = 0.001`).
- Return value: dict with at least `loss`.

The architecture, initial training, forget split, and evaluation probes are fixed.

## Evaluation
Benchmarks:

- `resnet20-cifar10-class0`: ResNet-20 on CIFAR-10, forgetting class 0.
- `vgg16bn-cifar100-class0`: VGG-16-BN on CIFAR-100, forgetting class 0.
- `mobilenetv2-fmnist-class0`: MobileNetV2 on FashionMNIST, forgetting class 0.

Reported metrics:
- `retain_acc`: accuracy on non-forget test data.
- `forget_acc`: accuracy on forget-class test data (lower is better).
- `forget_mia_auc`: membership inference attack AUC on forget set (lower is better).
- `unlearn_score`: `(retain_acc + (1 - forget_acc) + (1 - forget_mia_auc)) / 3`.

Primary metric: `unlearn_score` (higher is better).

## Baselines
The baselines below run inside the same harness via edit ops; defaults follow the corresponding papers:

- `retain_finetune`: continue training only on retained data with the supplied Adam optimizer.
- `negative_gradient`: NegGrad-style ascent on forget loss combined with descent on retain loss (Golatkar et al., CVPR 2020, arXiv:1911.04933).
- `bad_teacher`: incompetent-teacher distillation forgetting (Chundawat et al., AAAI 2023, arXiv:2205.08096). Reference code: https://github.com/vikram2000b/bad-teaching-unlearning.
- `scrub`: SCRUB selective student-teacher scrubbing (Kurmanji et al., NeurIPS 2023, arXiv:2302.09880).
