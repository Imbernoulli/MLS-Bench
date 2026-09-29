"""Score spec for security-adversarial-training."""
from mlsbench.scoring.dsl import *

# Values are fractions [0,1] -> bound=1.0 (not 100.0)
# clean_acc, robust_acc_fgsm, robust_acc_pgd: all higher better, bounded [0,1]
# Settings match config labels: SmallCNN-MNIST, PreActResNet18-C10, PreActResNet18-C100

# Robustness gate. Clean accuracy is a third of each setting, so a model trained
# without any adversarial training (PGD-50 robust accuracy ~0, clean accuracy above
# every defended baseline) used to score close to the published defenses. A setting
# whose PGD-50 robust accuracy is below half that of the weakest reference baseline
# is not an adversarially robust model: its score is multiplied by
# exp(-30 * (target - robust_acc_pgd)). Targets (half the weakest baseline):
# MNIST 0.44 (pgdat 0.8827), ResNet-C10 0.21 (trades 0.43), VGG-C10 0.19
# (awp 0.3759), ResNet-C100 0.10 (trades 0.1993). Every baseline clears them.
_PGD_GATE = {
    "SmallCNN_MNIST": 0.44,
    "PreActResNet18_C10": 0.21,
    "VGG11BN_C10": 0.19,
    "PreActResNet18_C100": 0.10,
}
for _slug, _target in _PGD_GATE.items():
    term(f"pgd_gate_{_slug}",
        penalty_lower(col(f"robust_acc_pgd_{_slug}").higher().id(), target=_target, sharpness=30.0))

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
), constraints=["pgd_gate_SmallCNN_MNIST"])
setting("PreActResNet18-C10", weighted_mean(
    ("clean_acc_PreActResNet18_C10", 1.0),
    ("robust_acc_fgsm_PreActResNet18_C10", 1.0),
    ("robust_acc_pgd_PreActResNet18_C10", 1.0),
), constraints=["pgd_gate_PreActResNet18_C10"])
setting("VGG11BN-C10", weighted_mean(
    ("clean_acc_VGG11BN_C10", 1.0),
    ("robust_acc_fgsm_VGG11BN_C10", 1.0),
    ("robust_acc_pgd_VGG11BN_C10", 1.0),
), constraints=["pgd_gate_VGG11BN_C10"])
setting("PreActResNet18-C100", weighted_mean(
    ("clean_acc_PreActResNet18_C100", 1.0),
    ("robust_acc_fgsm_PreActResNet18_C100", 1.0),
    ("robust_acc_pgd_PreActResNet18_C100", 1.0),
), constraints=["pgd_gate_PreActResNet18_C100"])

task(gmean("SmallCNN-MNIST", "PreActResNet18-C10", "VGG11BN-C10", "PreActResNet18-C100"))
