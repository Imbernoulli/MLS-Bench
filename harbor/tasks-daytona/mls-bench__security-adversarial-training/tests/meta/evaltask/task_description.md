# Adversarial Training for Model Robustness

## Research Question
How can we design better adversarial training methods to enhance model robustness against `L_inf` adversarial attacks?

## Background
Adversarial training is the most effective approach for improving neural-network robustness against adversarial examples. The standard method (Madry et al., 2018, arXiv:1706.06083) trains on PGD-generated adversarial examples using cross-entropy loss but suffers from a tradeoff between clean accuracy and robust accuracy. Advanced methods such as TRADES (Zhang et al., ICML 2019, arXiv:1901.08573) and MART (Wang et al., ICLR 2020, OpenReview rklOg6EFwS) address this through different loss formulations that decouple the robustness objective from clean classification. AWP (Wu, Xia, Wang, NeurIPS 2020, arXiv:2004.05884) further regularizes the flatness of the weight loss landscape and combines naturally with TRADES-style training.

## Task
Implement a novel adversarial training method in `torchattacks/bench/custom_adv_train.py` by modifying the `AdversarialTrainer` class. Your method should improve robust accuracy against white-box `L_inf` attacks while maintaining reasonable clean accuracy.

## Editable Interface
You must implement the `AdversarialTrainer` class with two methods:

```python
class AdversarialTrainer:
    def __init__(self, model, eps, alpha, attack_steps, num_classes, **kwargs):
        ...

    def train_step(self, images, labels, optimizer) -> dict:
        ...
```

`__init__`:
- `model`: the neural network to train (`nn.Module`).
- `eps`: `L_inf` perturbation budget (`0.3` for MNIST, `8/255` for CIFAR).
- `alpha`: step size for the inner PGD attack.
- `attack_steps`: number of PGD steps for adversarial example generation.
- `num_classes`: number of output classes (10 or 100).

`train_step`:
- `images`: clean images, shape `(N, C, H, W)`, values in `[0, 1]`.
- `labels`: ground-truth labels, shape `(N,)`.
- `optimizer`: SGD optimizer (`lr`, `momentum`, `weight_decay` already configured).
- Returns: dict with at least a `'loss'` key (float).

The training loop, learning-rate schedule (cosine annealing), model architecture, and data loading are handled externally. You only control the adversarial training procedure within each step.

## Evaluation
After training, models are evaluated on:
- **Clean accuracy**: accuracy on unperturbed test images.
- **Robust accuracy (FGSM)**: accuracy under one-step FGSM attack.
- **Robust accuracy (PGD-50)**: accuracy under a 50-step PGD attack — primary metric.

Each attack is run with two losses, cross-entropy and the Carlini-Wagner logit margin (`max_{j != y} z_j - z_y`, which does not depend on the logit scale), and a test image counts as robust only if it survives both. Evaluation runs in a separate process after training has exited: the harness builds a fresh model of the fixed architecture, loads only the trained `state_dict` into it (strict key and shape match), and evaluates it in eval mode, so anything other than the trained weights (hooks, patched methods or functions, wrapper modules) is not part of the evaluated model.

Scenarios (model + dataset):
- SmallCNN on MNIST (`eps = 0.3`)
- PreActResNet-18 on CIFAR-10 (`eps = 8/255`)
- VGG-11-BN on CIFAR-10 (`eps = 8/255`)
- PreActResNet-18 on CIFAR-100 (`eps = 8/255`)

Higher robust accuracy under PGD-50 across all scenarios is better. A scenario whose PGD-50 robust accuracy falls below half that of the weakest reference baseline (MNIST 0.44, PreActResNet18-C10 0.21, VGG11BN-C10 0.19, PreActResNet18-C100 0.10) does not count as adversarially robust: its score is multiplied by `exp(-30 * shortfall)`.

## Baselines
The baselines below run inside the same harness via edit ops; defaults follow the corresponding papers:

- `pgdat`: PGD adversarial training (Madry et al., 2018, arXiv:1706.06083). Standard PGD inner attack, cross-entropy loss on adversarial examples.
- `trades`: TRADES (Zhang et al., ICML 2019, arXiv:1901.08573). Cross-entropy on clean inputs plus a KL-divergence robustness regularizer with default `beta = 6.0` from the paper. Reference code: https://github.com/yaodongyu/TRADES.
- `mart`: MART (Wang et al., ICLR 2020). Misclassification-aware regularization that focuses on hard examples; default loss weight `lambda = 5.0` from the paper. Reference code: https://github.com/YisenWang/MART.
- `awp`: AWP combined with TRADES (Wu, Xia, Wang, NeurIPS 2020, arXiv:2004.05884). Adversarial weight perturbation on top of TRADES with default weight-perturbation magnitude `gamma = 5e-3` from the paper. Reference code: https://github.com/csdongxian/AWP.
