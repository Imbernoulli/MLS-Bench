"""Score spec for ai4bio-protein-inverse-folding (auto-generated, review before use)."""
from mlsbench.scoring.dsl import *

term("recovery_CATH4_2",
    col("recovery_CATH4.2").higher().id()
    .bounded_power(bound=1.0))

term("perplexity_CATH4_2",
    col("perplexity_CATH4.2").lower().id()
    .bounded_power(bound=1.0))

term("recovery_CATH4_3",
    col("recovery_CATH4.3").higher().id()
    .bounded_power(bound=1.0))

term("perplexity_CATH4_3",
    col("perplexity_CATH4.3").lower().id()
    .bounded_power(bound=1.0))

term("recovery_TS50",
    col("recovery_TS50").higher().id()
    .bounded_power(bound=1.0))

term("perplexity_TS50",
    col("perplexity_TS50").lower().id()
    .bounded_power(bound=1.0))

setting("CATH4.2", weighted_mean(("recovery_CATH4_2", 1.0), ("perplexity_CATH4_2", 1.0)))
setting("CATH4.3", weighted_mean(("recovery_CATH4_3", 1.0), ("perplexity_CATH4_3", 1.0)))
setting("TS50", weighted_mean(("recovery_TS50", 1.0), ("perplexity_TS50", 1.0)))

task(gmean("CATH4.2", "CATH4.3", "TS50"))
