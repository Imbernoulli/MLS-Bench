"""Score spec for dl-normalization (auto-generated, review before use)."""
from mlsbench.scoring.dsl import *

term("test_acc_resnet56_cifar100",
    col("test_acc_resnet56-cifar100").higher().id()
    .bounded_power(bound=100.0))

term("test_acc_resnet110_cifar100",
    col("test_acc_resnet110-cifar100").higher().id()
    .bounded_power(bound=100.0))

term("test_acc_mobilenetv2_fmnist",
    col("test_acc_mobilenetv2-fmnist").higher().id()
    .bounded_power(bound=100.0))

setting("resnet56-cifar100", weighted_mean(("test_acc_resnet56_cifar100", 1.0)))
setting("resnet110-cifar100", weighted_mean(("test_acc_resnet110_cifar100", 1.0)))
setting("mobilenetv2-fmnist", weighted_mean(("test_acc_mobilenetv2_fmnist", 1.0)))

task(gmean("resnet56-cifar100", "resnet110-cifar100", "mobilenetv2-fmnist"))
