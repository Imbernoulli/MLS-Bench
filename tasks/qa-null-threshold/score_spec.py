"""Direct official-F1 scoring; requires the integrated floor-aware scorer."""
from mlsbench.scoring.dsl import *

term("f1_part0",
    col("f1_part0").higher().id()
    .bounded_power(bound=100.0, floor=const(0.0),
                   ref=const(50.0), ref_score=0.5))

term("f1_part1",
    col("f1_part1").higher().id()
    .bounded_power(bound=100.0, floor=const(0.0),
                   ref=const(50.0), ref_score=0.5))

term("f1_part2",
    col("f1_part2").higher().id()
    .bounded_power(bound=100.0, floor=const(0.0),
                   ref=const(50.0), ref_score=0.5))

setting("part0", weighted_mean(("f1_part0", 1.0)))
setting("part1", weighted_mean(("f1_part1", 1.0)))
setting("part2", weighted_mean(("f1_part2", 1.0)))

task(gmean('part0', 'part1', 'part2'))
