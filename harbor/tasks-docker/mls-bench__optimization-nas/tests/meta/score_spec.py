"""Score spec for optimization-nas (sample-efficient K=30 regime).

Anchors. Every term's floor is the weakest baseline for that metric
(``BaselineAnchors.worst_for``). Each evaluation draws a hidden random
relabeling of the operations (see scripts/nas_oracle_entry.py), so the same
search code reaches different cells in different runs, and the scored
quantity -- the test accuracy averaged over the five seeds -- varies from one
verification to the next. The three baselines differ by less than that noise
(leaderboard.csv holds each baseline's exact expectation over the 24
relabelings), so calibrating on their spread (the old ``bounded_power`` with
best baseline = 0.5) made the score hinge on noise.
Every term therefore uses a fixed sigmoid scale, set so that a stated gain
over the weakest baseline scores 0.5 (2 * sigmoid(gain / scale) - 1 = 0.5,
i.e. scale = gain / ln 3). Each gain is about four run-to-run standard
deviations sigma of the five-seed mean over random relabelings (exact, from
all 24 relabelings per seed; root mean square over the three baselines):

- CIFAR-10:       sigma = 0.20 pt, gain = +0.8 pt -> 0.5.
- CIFAR-100:      sigma = 0.41 pt, gain = +1.7 pt -> 0.5.
- ImageNet16-120: sigma = 0.45 pt, gain = +1.8 pt -> 0.5.

A 1-sigma difference moves a term by at most about 0.14, and real gains
still register.
"""
import math

from mlsbench.scoring.dsl import *

# 2 * sigmoid(gain / scale) - 1 = 0.5  <=>  scale = gain / ln 3.
SCALE_CIFAR_10 = 0.8 / math.log(3.0)
SCALE_CIFAR_100 = 1.7 / math.log(3.0)
SCALE_IMAGENET16_120 = 1.8 / math.log(3.0)

term("test_accuracy_CIFAR_10",
    col("test_accuracy_CIFAR-10").higher().id()
    .sigmoid(scale=SCALE_CIFAR_10))

term("test_accuracy_CIFAR_100",
    col("test_accuracy_CIFAR-100").higher().id()
    .sigmoid(scale=SCALE_CIFAR_100))

term("test_accuracy_ImageNet16_120",
    col("test_accuracy_ImageNet16-120").higher().id()
    .sigmoid(scale=SCALE_IMAGENET16_120))

setting("CIFAR-10", weighted_mean(("test_accuracy_CIFAR_10", 1.0)))
setting("CIFAR-100", weighted_mean(("test_accuracy_CIFAR_100", 1.0)))
setting("ImageNet16-120", weighted_mean(("test_accuracy_ImageNet16_120", 1.0)))

task(gmean("CIFAR-10", "CIFAR-100", "ImageNet16-120"))
