"""Score spec for optimization-parity.

One term per setting: mean held-out ``test_accuracy`` (``score`` is the same
number emitted twice by the parser, so it is not scored separately).

Accuracy is bounded in [0.5, 1]: 0.5 is chance for balanced parity labels and
1.0 is a perfectly learned secret. Every setting's worst baseline sits at
chance (n32-k8 multi_epoch 0.506, n50-k8 0.498, n64-k8 0.501), so the floor
(worst baseline) is chance to within 0.007. ``ref=0.75`` with the default
``ref_score=0.5`` pins the curve's midpoint halfway between chance and a
perfect solve, so gamma ~= 1 (0.98-1.01): the term is ~linear from chance
(0) to perfect (1) and does NOT recalibrate on the best baseline. On n50-k8
and n64-k8 every baseline is at chance, and calibrating on the best one there
produced a tiny sigmoid scale that let +-0.002 of chance-level noise swing a
setting between 0 and ~0.9.

A setting at chance scores <~0.006 (0.003 accuracy noise / 0.5 range), below
the gmean epsilon (0.01), so it contributes the constant 0.01 whatever the
noise: chance-level settings neither zero the task nor reorder methods, while
real progress past chance on n50/n64 is rewarded strongly.
"""
from mlsbench.scoring.dsl import *

term("test_accuracy_n32_k8",
    col("test_accuracy_n32-k8").higher().id()
    .bounded_power(bound=1.0, ref=0.75))

term("test_accuracy_n50_k8",
    col("test_accuracy_n50-k8").higher().id()
    .bounded_power(bound=1.0, ref=0.75))

term("test_accuracy_n64_k8",
    col("test_accuracy_n64-k8").higher().id()
    .bounded_power(bound=1.0, ref=0.75))

setting("n32-k8", weighted_mean(("test_accuracy_n32_k8", 1.0)))
setting("n50-k8", weighted_mean(("test_accuracy_n50_k8", 1.0)))
setting("n64-k8", weighted_mean(("test_accuracy_n64_k8", 1.0)))

task(gmean("n32-k8", "n50-k8", "n64-k8"))
