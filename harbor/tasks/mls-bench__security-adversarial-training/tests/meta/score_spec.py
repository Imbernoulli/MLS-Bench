"""Score spec for security-adversarial-training."""
from mlsbench.scoring.dsl import *

# Values are fractions [0,1] -> bound=1.0 (not 100.0)
# clean_acc, robust_acc_fgsm, robust_acc_pgd: all higher better, bounded [0,1]
# Settings match config labels: SmallCNN-MNIST, PreActResNet18-C10, PreActResNet18-C100

term("clean_acc_SmallCNN_MNIST",
    col("clean_acc_SmallCNN_MNIST").higher().id()
    .bounded_power(bound=1.0))

term("robust_acc_fgsm_SmallCNN_MNIST",
    col("robust_acc_fgsm_SmallCNN_MNIST").higher().id()
    .bounded_power(bound=1.0))

term("robust_acc_pgd_SmallCNN_MNIST",
    col("robust_acc_pgd_SmallCNN_MNIST").higher().id()
    .bounded_power(bound=1.0))

term("clean_acc_PreActResNet18_C10",
    col("clean_acc_PreActResNet18_C10").higher().id()
    .bounded_power(bound=1.0))

term("robust_acc_fgsm_PreActResNet18_C10",
    col("robust_acc_fgsm_PreActResNet18_C10").higher().id()
    .bounded_power(bound=1.0))

term("robust_acc_pgd_PreActResNet18_C10",
    col("robust_acc_pgd_PreActResNet18_C10").higher().id()
    .bounded_power(bound=1.0))

term("clean_acc_VGG11BN_C10",
    col("clean_acc_VGG11BN_C10").higher().id()
    .bounded_power(bound=1.0))

term("robust_acc_fgsm_VGG11BN_C10",
    col("robust_acc_fgsm_VGG11BN_C10").higher().id()
    .bounded_power(bound=1.0))

term("robust_acc_pgd_VGG11BN_C10",
    col("robust_acc_pgd_VGG11BN_C10").higher().id()
    .bounded_power(bound=1.0))

term("clean_acc_PreActResNet18_C100",
    col("clean_acc_PreActResNet18_C100").higher().id()
    .bounded_power(bound=1.0))

term("robust_acc_fgsm_PreActResNet18_C100",
    col("robust_acc_fgsm_PreActResNet18_C100").higher().id()
    .bounded_power(bound=1.0))

term("robust_acc_pgd_PreActResNet18_C100",
    col("robust_acc_pgd_PreActResNet18_C100").higher().id()
    .bounded_power(bound=1.0))

setting("SmallCNN-MNIST", weighted_mean(
    ("clean_acc_SmallCNN_MNIST", 1.0),
    ("robust_acc_fgsm_SmallCNN_MNIST", 1.0),
    ("robust_acc_pgd_SmallCNN_MNIST", 1.0),
))
setting("PreActResNet18-C10", weighted_mean(
    ("clean_acc_PreActResNet18_C10", 1.0),
    ("robust_acc_fgsm_PreActResNet18_C10", 1.0),
    ("robust_acc_pgd_PreActResNet18_C10", 1.0),
))
setting("VGG11BN-C10", weighted_mean(
    ("clean_acc_VGG11BN_C10", 1.0),
    ("robust_acc_fgsm_VGG11BN_C10", 1.0),
    ("robust_acc_pgd_VGG11BN_C10", 1.0),
))
setting("PreActResNet18-C100", weighted_mean(
    ("clean_acc_PreActResNet18_C100", 1.0),
    ("robust_acc_fgsm_PreActResNet18_C100", 1.0),
    ("robust_acc_pgd_PreActResNet18_C100", 1.0),
))

task(gmean("SmallCNN-MNIST", "PreActResNet18-C10", "VGG11BN-C10", "PreActResNet18-C100"))
