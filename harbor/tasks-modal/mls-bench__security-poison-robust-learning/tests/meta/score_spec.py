"""Score spec for security-poison-robust-learning."""
from mlsbench.scoring.dsl import *

# test_acc: values are fractions [0,1] -> bound=1.0 (not 100.0)
# poison_fit: lower better (want model to resist poisoning), bounded [0,1]
# robust_score: higher better (composite robustness metric), bounded [0,1]

term("test_acc_vgg16bn_cifar100_labelflip",
    col("test_acc_vgg16bn_cifar100_labelflip").higher().id()
    .bounded_power(bound=1.0))

term("poison_fit_vgg16bn_cifar100_labelflip",
    col("poison_fit_vgg16bn_cifar100_labelflip").lower().id()
    .bounded_power(bound=0.0))

term("robust_score_vgg16bn_cifar100_labelflip",
    col("robust_score_vgg16bn_cifar100_labelflip").higher().id()
    .bounded_power(bound=1.0))

term("test_acc_resnet20_cifar10_labelflip",
    col("test_acc_resnet20_cifar10_labelflip").higher().id()
    .bounded_power(bound=1.0))

term("poison_fit_resnet20_cifar10_labelflip",
    col("poison_fit_resnet20_cifar10_labelflip").lower().id()
    .bounded_power(bound=0.0))

term("robust_score_resnet20_cifar10_labelflip",
    col("robust_score_resnet20_cifar10_labelflip").higher().id()
    .bounded_power(bound=1.0))

term("test_acc_mobilenetv2_fmnist_labelflip",
    col("test_acc_mobilenetv2_fmnist_labelflip").higher().id()
    .bounded_power(bound=1.0))

term("poison_fit_mobilenetv2_fmnist_labelflip",
    col("poison_fit_mobilenetv2_fmnist_labelflip").lower().id()
    .bounded_power(bound=0.0))

term("robust_score_mobilenetv2_fmnist_labelflip",
    col("robust_score_mobilenetv2_fmnist_labelflip").higher().id()
    .bounded_power(bound=1.0))

setting("vgg16bn-cifar100-labelflip", weighted_mean(
    ("test_acc_vgg16bn_cifar100_labelflip", 1.0),
    ("poison_fit_vgg16bn_cifar100_labelflip", 1.0),
    ("robust_score_vgg16bn_cifar100_labelflip", 1.0),
))
setting("resnet20-cifar10-labelflip", weighted_mean(
    ("test_acc_resnet20_cifar10_labelflip", 1.0),
    ("poison_fit_resnet20_cifar10_labelflip", 1.0),
    ("robust_score_resnet20_cifar10_labelflip", 1.0),
))
setting("mobilenetv2-fmnist-labelflip", weighted_mean(
    ("test_acc_mobilenetv2_fmnist_labelflip", 1.0),
    ("poison_fit_mobilenetv2_fmnist_labelflip", 1.0),
    ("robust_score_mobilenetv2_fmnist_labelflip", 1.0),
))

task(gmean("vgg16bn-cifar100-labelflip", "resnet20-cifar10-labelflip", "mobilenetv2-fmnist-labelflip"))
