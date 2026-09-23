"""Score spec for cv-classification-loss.

Anchors. Every term's floor is the weakest baseline for that metric
(``BaselineAnchors.worst_for``). The baselines span only
0.31 pt (ResNet-56) and 0.61 pt (VGG-16-BN) on CIFAR-100 and 0.68 pt on
Fashion-MNIST, one to three run-to-run standard deviations, so calibrating each
term on that spread (the default, best baseline = 0.5) made the score hinge on
noise.
Every term therefore uses a fixed sigmoid scale, set so that a stated gain
over the weakest baseline scores 0.5 (2 * sigmoid(gain / scale) - 1 = 0.5,
i.e. scale = gain / ln 3):

- CIFAR-10 and Fashion-MNIST test accuracy: +1.0 point.
- CIFAR-100 test accuracy: +2.0 points.

Each gain is about four run-to-run standard deviations. The standard deviation
is the binomial sampling error of the 10,000-image test set: 0.25 pt on
CIFAR-10, 0.22 pt on Fashion-MNIST and 0.45 pt on CIFAR-100. It bounds the
spread measured when the strongest baseline of this task family was re-run at
full scale on other hardware: 0.10, 0.09-0.18 and 0.21-0.46 pt. A 1-sigma
difference moves a term by at most about 0.14, and real gains still register.
"""
import math

from mlsbench.scoring.dsl import *

# 2 * sigmoid(gain / scale) - 1 = 0.5  <=>  scale = gain / ln 3.
SCALE_ACC_1PT = 1.0 / math.log(3.0)  # CIFAR-10, Fashion-MNIST: +1.0 pt -> 0.5
SCALE_ACC_2PT = 2.0 / math.log(3.0)  # CIFAR-100: +2.0 pt -> 0.5

term("test_acc_resnet56_cifar100",
    col("test_acc_resnet56-cifar100").higher().id()
    .sigmoid(scale=SCALE_ACC_2PT))

term("test_acc_vgg16bn_cifar100",
    col("test_acc_vgg16bn-cifar100").higher().id()
    .sigmoid(scale=SCALE_ACC_2PT))

term("test_acc_mobilenetv2_fmnist",
    col("test_acc_mobilenetv2-fmnist").higher().id()
    .sigmoid(scale=SCALE_ACC_1PT))

setting("resnet56-cifar100", weighted_mean(("test_acc_resnet56_cifar100", 1.0)))
setting("vgg16bn-cifar100", weighted_mean(("test_acc_vgg16bn_cifar100", 1.0)))
setting("mobilenetv2-fmnist", weighted_mean(("test_acc_mobilenetv2_fmnist", 1.0)))

task(gmean("resnet56-cifar100", "vgg16bn-cifar100", "mobilenetv2-fmnist"))
