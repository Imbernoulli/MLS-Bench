"""Score spec for optimization-gradient-compression."""
from mlsbench.scoring.dsl import *

# test_acc and best_acc are nearly identical (final vs peak); keep best_acc as primary
# accuracy on 0-100 scale (values: 92.51, 70.77, 94.14)
# test_loss: lower is better, bound=0.0
# refs from best baseline means

term("best_acc_resnet20_cifar10",
    col("best_acc_resnet20-cifar10").higher().id()
    .bounded_power(bound=100.0))

term("test_loss_resnet20_cifar10",
    col("test_loss_resnet20-cifar10").lower().id()
    .bounded_power(bound=0.0))

term("best_acc_vgg11_cifar100",
    col("best_acc_vgg11-cifar100").higher().id()
    .bounded_power(bound=100.0))

term("test_loss_vgg11_cifar100",
    col("test_loss_vgg11-cifar100").lower().id()
    .bounded_power(bound=0.0))

term("best_acc_resnet56_cifar10",
    col("best_acc_resnet56-cifar10").higher().id()
    .bounded_power(bound=100.0))

term("test_loss_resnet56_cifar10",
    col("test_loss_resnet56-cifar10").lower().id()
    .bounded_power(bound=0.0))

setting("resnet20-cifar10", weighted_mean(
    ("best_acc_resnet20_cifar10", 1.0),
    ("test_loss_resnet20_cifar10", 1.0),
))
setting("vgg11-cifar100", weighted_mean(
    ("best_acc_vgg11_cifar100", 1.0),
    ("test_loss_vgg11_cifar100", 1.0),
))
setting("resnet56-cifar10", weighted_mean(
    ("best_acc_resnet56_cifar10", 1.0),
    ("test_loss_resnet56_cifar10", 1.0),
))

task(gmean("resnet20-cifar10", "vgg11-cifar100", "resnet56-cifar10"))
