"""Score spec for cv-multitask-loss (auto-generated, review before use)."""
from mlsbench.scoring.dsl import *

term("test_acc_resnet20_cifar100mt",
    col("test_acc_resnet20-cifar100mt").higher().id()
    .bounded_power(bound=100.0))

term("test_acc_resnet56_cifar100mt",
    col("test_acc_resnet56-cifar100mt").higher().id()
    .bounded_power(bound=100.0))

term("test_acc_vgg16bn_cifar100mt",
    col("test_acc_vgg16bn-cifar100mt").higher().id()
    .bounded_power(bound=100.0))

setting("resnet20-cifar100mt", weighted_mean(("test_acc_resnet20_cifar100mt", 1.0)))
setting("resnet56-cifar100mt", weighted_mean(("test_acc_resnet56_cifar100mt", 1.0)))
setting("vgg16bn-cifar100mt", weighted_mean(("test_acc_vgg16bn_cifar100mt", 1.0)))

task(gmean("resnet20-cifar100mt", "resnet56-cifar100mt", "vgg16bn-cifar100mt"))
