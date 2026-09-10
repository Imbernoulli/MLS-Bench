"""Score spec for llm-ptq-algorithm (auto-generated)."""
from mlsbench.scoring.dsl import *

term("wikitext2_ppl_ptq_7b_int4",
    col("wikitext2_ppl_ptq-7b-int4").lower().id()
    .bounded_power(bound=1.0))

term("degradation_ptq_7b_int4",
    col("degradation_ptq-7b-int4").lower().id()
    .bounded_power(bound=0.0))

term("wikitext2_ppl_ptq_7b_int3",
    col("wikitext2_ppl_ptq-7b-int3").lower().id()
    .bounded_power(bound=1.0))

term("degradation_ptq_7b_int3",
    col("degradation_ptq-7b-int3").lower().id()
    .bounded_power(bound=0.0))

term("wikitext2_ppl_ptq_7b_int4_g64",
    col("wikitext2_ppl_ptq-7b-int4-g64").lower().id()
    .bounded_power(bound=1.0))

term("degradation_ptq_7b_int4_g64",
    col("degradation_ptq-7b-int4-g64").lower().id()
    .bounded_power(bound=0.0))

setting("ptq-7b-int4", weighted_mean(("wikitext2_ppl_ptq_7b_int4", 1.0), ("degradation_ptq_7b_int4", 1.0)))
setting("ptq-7b-int3", weighted_mean(("wikitext2_ppl_ptq_7b_int3", 1.0), ("degradation_ptq_7b_int3", 1.0)))
setting("ptq-7b-int4-g64", weighted_mean(("wikitext2_ppl_ptq_7b_int4_g64", 1.0), ("degradation_ptq_7b_int4_g64", 1.0)))

task(gmean("ptq-7b-int4", "ptq-7b-int3", "ptq-7b-int4-g64"))
