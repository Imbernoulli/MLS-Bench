"""Score spec for security-adversarial-attack-sparse-l0.

ATTACKER task: ASR = attack success rate against an adversarially-robust
target (== robust error in Croce et al., AAAI 2022). Higher is better for
the attacker, bounded [0, 1].

Canonical Sparse-RS L0 setting (arXiv:2006.12834v3, Table 2 / App. A.5):
k = 24 perturbed pixels, untargeted, RobustBench L2-robust CIFAR-10 models.
On robust targets Sparse-RS lands well below 1.0 (paper: 81-86% robust
error), so the worst-baseline floor / best-baseline anchor stay separated
and the task is not saturated.

Settings match config labels: Rebuffi-R18-L2, Augustin-L2, Engstrom-L2.
"""
from mlsbench.scoring.dsl import *

term("asr_Rebuffi_R18_L2",
    col("asr_Rebuffi_R18_L2").higher().id()
    .bounded_power(bound=1.0))

term("asr_Augustin_L2",
    col("asr_Augustin_L2").higher().id()
    .bounded_power(bound=1.0))

term("asr_Engstrom_L2",
    col("asr_Engstrom_L2").higher().id()
    .bounded_power(bound=1.0))

setting("Rebuffi-R18-L2", weighted_mean(("asr_Rebuffi_R18_L2", 1.0)))
setting("Augustin-L2", weighted_mean(("asr_Augustin_L2", 1.0)))
setting("Engstrom-L2", weighted_mean(("asr_Engstrom_L2", 1.0)))

task(gmean("Rebuffi-R18-L2", "Augustin-L2", "Engstrom-L2"))
