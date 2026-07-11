"""Three-setting score spec for the full-image logit-only OOD protocol.

All anchors were measured on one H20 with the pinned OpenOOD-style CIFAR-10
ResNet-18 checkpoint (full-test ID accuracy 0.9514). MSP is the native weak
surface. The ID-only pseudo-class cosine-prototype scorer is the strong anchor
because it improves AUROC over MSP on every setting:

  setting       MSP          pseudo-cosine
  SVHN          0.92846834   0.93220756
  CIFAR-100     0.87088166   0.89527483
  Tiny-ImageNet 0.87007299   0.89377779

Each sigmoid uses pseudo-cosine as its 0.5 reference and
``(pseudo_cosine - MSP) / ln(9)`` as its scale, mapping the measured MSP value
to 0.1 without a hard floor. Every setting contributes through task gmean.
"""
from mlsbench.scoring.dsl import *


term(
    "auroc_ood_logit_svhn_full",
    col("auroc_ood_logit_svhn_full").higher().id().sigmoid(
        ref=const(0.93220756), scale=0.001701792360493821,
    ),
)
term(
    "auroc_ood_logit_cifar100_full",
    col("auroc_ood_logit_cifar100_full").higher().id().sigmoid(
        ref=const(0.89527483), scale=0.011101810097888468,
    ),
)
term(
    "auroc_ood_logit_tin_full",
    col("auroc_ood_logit_tin_full").higher().id().sigmoid(
        ref=const(0.89377779), scale=0.010788519409671913,
    ),
)

setting(
    "ood_logit_svhn_full",
    weighted_mean(("auroc_ood_logit_svhn_full", 1.0)),
)
setting(
    "ood_logit_cifar100_full",
    weighted_mean(("auroc_ood_logit_cifar100_full", 1.0)),
)
setting(
    "ood_logit_tin_full",
    weighted_mean(("auroc_ood_logit_tin_full", 1.0)),
)
task(gmean("ood_logit_svhn_full", "ood_logit_cifar100_full", "ood_logit_tin_full"))
