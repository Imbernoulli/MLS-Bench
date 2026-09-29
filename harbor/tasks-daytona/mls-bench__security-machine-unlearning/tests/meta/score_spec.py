"""Score spec for security-machine-unlearning.

Class-wise forgetting. The gold standard is a model retrained from scratch
without the forget class: it never predicts that class (forget_acc 0) and its
membership-inference AUC on the forget class is chance (~0.5; measured 0.48 /
0.49 / 0.52 for a local 80-epoch retrain of the three settings). What an
unlearning rule can still do better or worse is keep the utility on the
retained classes, so per setting:

  score = retain_acc term  x  penalty(forget_acc <= 0.01)
                           x  penalty(|forget_mia_auc - 0.5| <= 0.06)

- retain_acc: higher is better, bound 1.0. Its floor is the worst baseline,
  which is negative_gradient at chance (1/99, 0.17, 1/9) -- the natural
  "destroyed model" floor, so a constant predictor scores 0 here. (The two
  collapsed anchors are kept for this column on purpose: without them the
  floor would sit 1-5 points under the best baseline, inside seed noise.)
- forget_acc is a constraint, not an objective: every non-collapsed baseline
  and the retrain reference sit at 0, so as a bounded term it was a step
  function that handed a destroyed model 1/3 of each setting for free and
  gave SCRUB's 0.005 a 0.
- forget_mia_gap = |forget_mia_auc - 0.5| (computed by parser.py) is a
  constraint as well. An AUC below 0.5 separates members from non-members as
  well as one above it, so the old 1 - auc reward is gone. The tolerance 0.06
  is ~2 standard errors of the AUC on the smallest probe (CIFAR-100: 500
  members vs 100 non-members); baselines sit at 0.0005-0.058, the retrain
  reference at 0.014-0.018, the untouched original model at 0.02-0.07.
- unlearn_score is reported for reference only (it duplicates the above).
"""
from mlsbench.scoring.dsl import *

FORGET_ACC_MAX = 0.01
MIA_GAP_MAX = 0.06


def _add_setting(label):
    s = label.replace("-", "_")
    term(f"retain_acc_{s}",
        col(f"retain_acc_{s}").higher().id()
        .bounded_power(bound=1.0))
    term(f"forget_acc_max_{s}",
        penalty_upper(col(f"forget_acc_{s}").lower().id(),
                      target=FORGET_ACC_MAX, sharpness=20.0))
    term(f"forget_mia_gap_max_{s}",
        penalty_upper(col(f"forget_mia_gap_{s}").lower().id(),
                      target=MIA_GAP_MAX, sharpness=20.0))
    setting(label, weighted_mean(
        (f"retain_acc_{s}", 1.0),
    ), constraints=[f"forget_acc_max_{s}", f"forget_mia_gap_max_{s}"])


for _label in ("vgg16bn-cifar100-class0", "resnet20-cifar10-class0",
               "mobilenetv2-fmnist-class0"):
    _add_setting(_label)

task(gmean("vgg16bn-cifar100-class0", "resnet20-cifar10-class0", "mobilenetv2-fmnist-class0"))
