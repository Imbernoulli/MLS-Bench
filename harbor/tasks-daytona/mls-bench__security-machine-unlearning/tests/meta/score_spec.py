"""Score spec for security-machine-unlearning."""
from mlsbench.scoring.dsl import *

# retain_acc: higher better (maintain performance on retained data), values in [0,1] -> bound=1.0
# forget_acc: LOWER better (want model to forget, so low accuracy on forgotten class)
# forget_mia_auc is reported for diagnostics only. The target is random guessing
# around 0.5, so raw AUC cannot be scored as lower-is-better toward 0.0.
# unlearn_score: higher better (composite metric), bounded [0,1]

term("retain_acc_vgg16bn_cifar100_class0",
    col("retain_acc_vgg16bn_cifar100_class0").higher().id()
    .bounded_power(bound=1.0))

term("forget_acc_vgg16bn_cifar100_class0",
    col("forget_acc_vgg16bn_cifar100_class0").lower().id()
    .bounded_power(bound=0.0))

term("unlearn_score_vgg16bn_cifar100_class0",
    col("unlearn_score_vgg16bn_cifar100_class0").higher().id()
    .bounded_power(bound=1.0))

term("retain_acc_resnet20_cifar10_class0",
    col("retain_acc_resnet20_cifar10_class0").higher().id()
    .bounded_power(bound=1.0))

term("forget_acc_resnet20_cifar10_class0",
    col("forget_acc_resnet20_cifar10_class0").lower().id()
    .bounded_power(bound=0.0))

term("unlearn_score_resnet20_cifar10_class0",
    col("unlearn_score_resnet20_cifar10_class0").higher().id()
    .bounded_power(bound=1.0))

term("retain_acc_mobilenetv2_fmnist_class0",
    col("retain_acc_mobilenetv2_fmnist_class0").higher().id()
    .bounded_power(bound=1.0))

term("forget_acc_mobilenetv2_fmnist_class0",
    col("forget_acc_mobilenetv2_fmnist_class0").lower().id()
    .bounded_power(bound=0.0))

term("unlearn_score_mobilenetv2_fmnist_class0",
    col("unlearn_score_mobilenetv2_fmnist_class0").higher().id()
    .bounded_power(bound=1.0))

setting("vgg16bn-cifar100-class0", weighted_mean(
    ("retain_acc_vgg16bn_cifar100_class0", 1.0),
    ("forget_acc_vgg16bn_cifar100_class0", 1.0),
    ("unlearn_score_vgg16bn_cifar100_class0", 1.0),
))
setting("resnet20-cifar10-class0", weighted_mean(
    ("retain_acc_resnet20_cifar10_class0", 1.0),
    ("forget_acc_resnet20_cifar10_class0", 1.0),
    ("unlearn_score_resnet20_cifar10_class0", 1.0),
))
setting("mobilenetv2-fmnist-class0", weighted_mean(
    ("retain_acc_mobilenetv2_fmnist_class0", 1.0),
    ("forget_acc_mobilenetv2_fmnist_class0", 1.0),
    ("unlearn_score_mobilenetv2_fmnist_class0", 1.0),
))

task(gmean("vgg16bn-cifar100-class0", "resnet20-cifar10-class0", "mobilenetv2-fmnist-class0"))
