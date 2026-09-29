"""Score spec for security-membership-inference-defense."""
from mlsbench.scoring.dsl import *

# Per setting: test_acc (utility), mia_adv = |mia_auc - 0.5| (attack
# advantage) and privacy_score = test_acc - |mia_auc - 0.5| (the primary
# metric), averaged, times a minimum-accuracy gate.
#
# - privacy_gap (member minus non-member mean max-softmax) is reported but not
#   scored: it depends on the logit scale, so a loss that only shrinks the
#   logits drives it to ~0 with accuracy and attack AUC unchanged, and a
#   constant predictor has gap 0. mia_adv replaces it as the privacy-only term.
# - mia_auc is scored as |mia_auc - 0.5|: an AUC below 0.5 leaks as much as one
#   above it (the attacker flips its rule).
# - The gate (0.8 x the ERM baseline's test_acc, rounded down) keeps an
#   uninformative predictor (constant or untrained, AUC 0.5) from buying
#   perfect privacy with no utility. Every baseline clears it.
_SETTINGS = {
    "resnet20-cifar10": ("resnet20_cifar10", 0.63),
    "vgg16bn-cifar100": ("vgg16bn_cifar100", 0.39),
    "mobilenetv2-fmnist": ("mobilenetv2_fmnist", 0.74),
}

for _label, (_sfx, _min_acc) in _SETTINGS.items():
    term(f"test_acc_{_sfx}",
        col(f"test_acc_{_sfx}").higher().id()
        .bounded_power(bound=1.0))
    term(f"mia_adv_{_sfx}",
        col(f"mia_adv_{_sfx}").lower().id()
        .bounded_power(bound=0.0))
    term(f"privacy_score_{_sfx}",
        col(f"privacy_score_{_sfx}").higher().id()
        .bounded_power(bound=1.0))
    term(f"test_acc_min_{_sfx}",
        penalty_lower(col(f"test_acc_{_sfx}").higher().id(),
                      target=_min_acc, sharpness=20.0))
    setting(_label, weighted_mean(
        (f"test_acc_{_sfx}", 1.0),
        (f"mia_adv_{_sfx}", 1.0),
        (f"privacy_score_{_sfx}", 1.0),
    ), constraints=[f"test_acc_min_{_sfx}"])

task(gmean("resnet20-cifar10", "vgg16bn-cifar100", "mobilenetv2-fmnist"))
