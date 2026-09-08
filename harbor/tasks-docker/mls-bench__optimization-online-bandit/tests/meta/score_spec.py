"""Score spec for optimization-online-bandit."""
from mlsbench.scoring.dsl import *

# normalized_regret: lower is better (fraction of optimal reward missed)
# cumulative_regret = normalized_regret × 10000 — redundant, dropped
term("normalized_regret_stochastic_mab",
    col("normalized_regret_stochastic_mab").lower().id()
    .bounded_power(bound=0.0))

term("normalized_regret_contextual",
    col("normalized_regret_contextual").lower().id()
    .bounded_power(bound=0.0))

term("normalized_regret_nonstationary",
    col("normalized_regret_nonstationary").lower().id()
    .bounded_power(bound=0.0))

setting("stochastic-mab", weighted_mean(("normalized_regret_stochastic_mab", 1.0)))
setting("contextual", weighted_mean(("normalized_regret_contextual", 1.0)))
setting("nonstationary", weighted_mean(("normalized_regret_nonstationary", 1.0)))

task(gmean("stochastic-mab", "contextual", "nonstationary"))
