"""Score spec for optimization-nas (sample-efficient K=30 regime).

Bounds are the actual maxima in the local NAS-Bench-201 pickle queried by the
harness (94.68 / 73.26 / 47.333333 across CIFAR-10, CIFAR-100,
ImageNet16-120).
"""
from mlsbench.scoring.dsl import *

term("test_accuracy_CIFAR_10",
    col("test_accuracy_CIFAR-10").higher().id()
    .bounded_power(bound=94.68))

term("test_accuracy_CIFAR_100",
    col("test_accuracy_CIFAR-100").higher().id()
    .bounded_power(bound=73.26))

term("test_accuracy_ImageNet16_120",
    col("test_accuracy_ImageNet16-120").higher().id()
    .bounded_power(bound=47.333333))

setting("CIFAR-10", weighted_mean(("test_accuracy_CIFAR_10", 1.0)))
setting("CIFAR-100", weighted_mean(("test_accuracy_CIFAR_100", 1.0)))
setting("ImageNet16-120", weighted_mean(("test_accuracy_ImageNet16_120", 1.0)))

task(gmean("CIFAR-10", "CIFAR-100", "ImageNet16-120"))
