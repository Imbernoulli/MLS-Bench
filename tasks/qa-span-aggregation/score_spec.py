"""Direct official-F1 scoring; requires the integrated floor-aware scorer."""
from mlsbench.scoring.dsl import *

term("f1_squad",
    col("f1_squad").higher().id()
    .bounded_power(bound=100.0, floor=const(0.0),
                   ref=const(50.0), ref_score=0.5))

term("f1_newsqa",
    col("f1_newsqa").higher().id()
    .bounded_power(bound=100.0, floor=const(0.0),
                   ref=const(50.0), ref_score=0.5))

term("f1_hotpotqa",
    col("f1_hotpotqa").higher().id()
    .bounded_power(bound=100.0, floor=const(0.0),
                   ref=const(50.0), ref_score=0.5))

term("f1_naturalq",
    col("f1_naturalq").higher().id()
    .bounded_power(bound=100.0, floor=const(0.0),
                   ref=const(50.0), ref_score=0.5))

setting("squad", weighted_mean(("f1_squad", 1.0)))
setting("newsqa", weighted_mean(("f1_newsqa", 1.0)))
setting("hotpotqa", weighted_mean(("f1_hotpotqa", 1.0)))
setting("naturalq", weighted_mean(("f1_naturalq", 1.0)))

task(gmean('squad', 'newsqa', 'hotpotqa', 'naturalq'))
