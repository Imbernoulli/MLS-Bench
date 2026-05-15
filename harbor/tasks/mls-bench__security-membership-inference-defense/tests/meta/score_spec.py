"""Score spec for security-membership-inference-defense."""
from mlsbench.scoring.dsl import *

# test_acc: higher better (maintain utility), values in [0,1] -> bound=1.0 (not 100.0)
# mia_auc is diagnostic only. The defense target is random guessing near 0.5,
# so raw AUC should not be scored as lower-is-better toward 0.0.
# privacy_gap: lower better, bounded [0,1]
# privacy_score: higher better, bounded [0,1]

term("test_acc_resnet20_cifar10",
    col("test_acc_resnet20_cifar10").higher().id()
    .bounded_power(bound=1.0))

term("privacy_gap_resnet20_cifar10",
    col("privacy_gap_resnet20_cifar10").lower().id()
    .bounded_power(bound=0.0))

term("privacy_score_resnet20_cifar10",
    col("privacy_score_resnet20_cifar10").higher().id()
    .bounded_power(bound=1.0))

term("test_acc_vgg16bn_cifar100",
    col("test_acc_vgg16bn_cifar100").higher().id()
    .bounded_power(bound=1.0))

term("privacy_gap_vgg16bn_cifar100",
    col("privacy_gap_vgg16bn_cifar100").lower().id()
    .bounded_power(bound=0.0))

term("privacy_score_vgg16bn_cifar100",
    col("privacy_score_vgg16bn_cifar100").higher().id()
    .bounded_power(bound=1.0))

term("test_acc_mobilenetv2_fmnist",
    col("test_acc_mobilenetv2_fmnist").higher().id()
    .bounded_power(bound=1.0))

term("privacy_gap_mobilenetv2_fmnist",
    col("privacy_gap_mobilenetv2_fmnist").lower().id()
    .bounded_power(bound=0.0))

term("privacy_score_mobilenetv2_fmnist",
    col("privacy_score_mobilenetv2_fmnist").higher().id()
    .bounded_power(bound=1.0))

setting("resnet20-cifar10", weighted_mean(
    ("test_acc_resnet20_cifar10", 1.0),
    ("privacy_gap_resnet20_cifar10", 1.0),
    ("privacy_score_resnet20_cifar10", 1.0),
))
setting("vgg16bn-cifar100", weighted_mean(
    ("test_acc_vgg16bn_cifar100", 1.0),
    ("privacy_gap_vgg16bn_cifar100", 1.0),
    ("privacy_score_vgg16bn_cifar100", 1.0),
))
setting("mobilenetv2-fmnist", weighted_mean(
    ("test_acc_mobilenetv2_fmnist", 1.0),
    ("privacy_gap_mobilenetv2_fmnist", 1.0),
    ("privacy_score_mobilenetv2_fmnist", 1.0),
))

task(gmean("resnet20-cifar10", "vgg16bn-cifar100", "mobilenetv2-fmnist"))
