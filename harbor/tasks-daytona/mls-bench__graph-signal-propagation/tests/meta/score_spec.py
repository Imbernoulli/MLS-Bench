"""Score spec for graph-signal-propagation."""
from mlsbench.scoring.dsl import *

# accuracy is on 0-1 scale based on leaderboard values (0.87, 0.80, etc.)
term("accuracy_cora",
    col("accuracy_cora").higher().id()
    .bounded_power(bound=1.0))

term("accuracy_citeseer",
    col("accuracy_citeseer").higher().id()
    .bounded_power(bound=1.0))

term("accuracy_texas",
    col("accuracy_texas").higher().id()
    .bounded_power(bound=1.0))

term("accuracy_cornell",
    col("accuracy_cornell").higher().id()
    .bounded_power(bound=1.0))

# std_ metrics are within-run variance of accuracy; informational — dropped

setting("cora", weighted_mean(("accuracy_cora", 1.0)))
setting("citeseer", weighted_mean(("accuracy_citeseer", 1.0)))
setting("texas", weighted_mean(("accuracy_texas", 1.0)))
setting("cornell", weighted_mean(("accuracy_cornell", 1.0)))

task(gmean("cora", "citeseer", "texas", "cornell"))
