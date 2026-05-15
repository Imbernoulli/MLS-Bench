"""Score spec for ai4bio-mutation-effect-prediction."""
from mlsbench.scoring.dsl import *

# Spearman correlation: range [-1, 1], higher is better, bounded at 1.0

term("spearman_BLAT_ECOLX",
    col("spearman_BLAT_ECOLX").higher().id()
    .bounded_power(bound=1.0))

term("spearman_ESTA_BACSU",
    col("spearman_ESTA_BACSU").higher().id()
    .bounded_power(bound=1.0))

term("spearman_RASH_HUMAN",
    col("spearman_RASH_HUMAN").higher().id()
    .bounded_power(bound=1.0))

setting("BLAT_ECOLX", weighted_mean(("spearman_BLAT_ECOLX", 1.0)))
setting("ESTA_BACSU", weighted_mean(("spearman_ESTA_BACSU", 1.0)))
setting("RASH_HUMAN", weighted_mean(("spearman_RASH_HUMAN", 1.0)))

task(gmean("BLAT_ECOLX", "ESTA_BACSU", "RASH_HUMAN"))
