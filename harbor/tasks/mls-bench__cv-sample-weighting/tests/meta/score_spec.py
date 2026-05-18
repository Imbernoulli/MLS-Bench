"""Score spec for cv-sample-weighting (auto-generated, review before use)."""
from mlsbench.scoring.dsl import *

term("test_acc_resnet32_cifar10lt",
    col("test_acc_resnet32-cifar10lt").higher().id()
    .bounded_power(bound=100.0))

term("test_acc_resnet32_cifar100lt",
    col("test_acc_resnet32-cifar100lt").higher().id()
    .bounded_power(bound=100.0))

term("test_acc_vgg16bn_cifar100lt",
    col("test_acc_vgg16bn-cifar100lt").higher().id()
    .bounded_power(bound=100.0))

setting("resnet32-cifar10lt", weighted_mean(("test_acc_resnet32_cifar10lt", 1.0)))
setting("resnet32-cifar100lt", weighted_mean(("test_acc_resnet32_cifar100lt", 1.0)))
setting("vgg16bn-cifar100lt", weighted_mean(("test_acc_vgg16bn_cifar100lt", 1.0)))

task(gmean("resnet32-cifar10lt", "resnet32-cifar100lt", "vgg16bn-cifar100lt"))
