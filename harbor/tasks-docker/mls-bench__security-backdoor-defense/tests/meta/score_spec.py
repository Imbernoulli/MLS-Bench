"""Score spec for security-backdoor-defense.

Backdoor defense task across three model-dataset-attack combinations:
  - resnet20-cifar10-badnets
  - vgg16bn-cifar100-blend
  - mobilenetv2-fmnist-badnets

Each setting produces four metrics:
  - clean_acc: higher is better, bounded in [0, 1] — clean test accuracy of the
    RETRAINED defended model (post-filter)
  - asr (attack success rate): LOWER is better, bounded in [0, 1] — ASR of the
    RETRAINED defended model on triggered test inputs
  - poison_recall: higher is better, bounded in [0, 1] — diagnostic: fraction of
    true poisoned samples removed at filter stage (reported, not part of
    defense_score)
  - defense_score: higher is better, bounded in [0, 1] — primary metric;
    defense_score = 0.5 * clean_acc + 0.5 * (1 - asr), following
    BackdoorBench/defense/spectral.py's retrain-based ACC+(1-ASR)/2 convention

defense_score is the primary metric and is weighted most heavily. The filter
ratio is 1.5 * poison_fraction per Tran et al. (2018, Sec. 4.1).
"""
from mlsbench.scoring.dsl import *

# ---- resnet20-cifar10-badnets ----
# Reference points use BackdoorBench's retrained-model ACC+(1-ASR)/2 scale.
# The exact clean/ASR tradeoff is calibrated by this task's local baselines.
term("defense_resnet",
    col("defense_score_resnet20_cifar10_badnets").higher().id()
    .bounded_power(bound=1.0))
term("clean_acc_resnet",
    col("clean_acc_resnet20_cifar10_badnets").higher().id()
    .bounded_power(bound=1.0))
term("asr_resnet",
    col("asr_resnet20_cifar10_badnets").lower().id()
    .bounded_power(bound=0.0))
term("poison_recall_resnet",
    col("poison_recall_resnet20_cifar10_badnets").higher().id()
    .bounded_power(bound=1.0))

setting("resnet20-cifar10-badnets", weighted_mean(
    ("defense_resnet", 3.0),
    ("clean_acc_resnet", 1.0),
    ("asr_resnet", 1.0),
    ("poison_recall_resnet", 0.5),
))

# ---- vgg16bn-cifar100-blend ----
# With poison_fraction=0.01 (target-class ~33% poisoned), per-class SVD-style
# defenses can operate in their intended regime.  Clean acc for VGG16-BN on
# CIFAR-100 ceiling is ~0.72; blend trigger is harder to unlearn than BadNets.
term("defense_vgg",
    col("defense_score_vgg16bn_cifar100_blend").higher().id()
    .bounded_power(bound=1.0))
term("clean_acc_vgg",
    col("clean_acc_vgg16bn_cifar100_blend").higher().id()
    .bounded_power(bound=1.0))
term("asr_vgg",
    col("asr_vgg16bn_cifar100_blend").lower().id()
    .bounded_power(bound=0.0))
term("poison_recall_vgg",
    col("poison_recall_vgg16bn_cifar100_blend").higher().id()
    .bounded_power(bound=1.0))

setting("vgg16bn-cifar100-blend", weighted_mean(
    ("defense_vgg", 3.0),
    ("clean_acc_vgg", 1.0),
    ("asr_vgg", 1.0),
    ("poison_recall_vgg", 0.5),
))

# ---- mobilenetv2-fmnist-badnets ----
term("defense_mobile",
    col("defense_score_mobilenetv2_fmnist_badnets").higher().id()
    .bounded_power(bound=1.0))
term("clean_acc_mobile",
    col("clean_acc_mobilenetv2_fmnist_badnets").higher().id()
    .bounded_power(bound=1.0))
term("asr_mobile",
    col("asr_mobilenetv2_fmnist_badnets").lower().id()
    .bounded_power(bound=0.0))
term("poison_recall_mobile",
    col("poison_recall_mobilenetv2_fmnist_badnets").higher().id()
    .bounded_power(bound=1.0))

setting("mobilenetv2-fmnist-badnets", weighted_mean(
    ("defense_mobile", 3.0),
    ("clean_acc_mobile", 1.0),
    ("asr_mobile", 1.0),
    ("poison_recall_mobile", 0.5),
))

# Task: geometric mean across model-dataset-attack settings
task(gmean("resnet20-cifar10-badnets", "vgg16bn-cifar100-blend", "mobilenetv2-fmnist-badnets"))
