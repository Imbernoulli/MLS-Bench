"""Score spec for causal-discovery-discrete."""
from mlsbench.scoring.dsl import *

# Only metrics from config labels: Cancer, Child, Alarm, Hailfinder, Win95pts
# Other graph metrics (Earthquake, Survey, Asia, Sachs, etc.) exist in leaderboard from
# historical runs but are not current config labels, so excluded here.

term("shd_Cancer",
    col("shd_Cancer").lower().id()
    .bounded_power(bound=0.0))

term("adj_precision_Cancer",
    col("adj_precision_Cancer").higher().id()
    .bounded_power(bound=1.0))

term("adj_recall_Cancer",
    col("adj_recall_Cancer").higher().id()
    .bounded_power(bound=1.0))

term("arrow_precision_Cancer",
    col("arrow_precision_Cancer").higher().id()
    .bounded_power(bound=1.0))

term("arrow_recall_Cancer",
    col("arrow_recall_Cancer").higher().id()
    .bounded_power(bound=1.0))

term("shd_Child",
    col("shd_Child").lower().id()
    .bounded_power(bound=0.0))

term("adj_precision_Child",
    col("adj_precision_Child").higher().id()
    .bounded_power(bound=1.0))

term("adj_recall_Child",
    col("adj_recall_Child").higher().id()
    .bounded_power(bound=1.0))

term("arrow_precision_Child",
    col("arrow_precision_Child").higher().id()
    .bounded_power(bound=1.0))

term("arrow_recall_Child",
    col("arrow_recall_Child").higher().id()
    .bounded_power(bound=1.0))

term("shd_Alarm",
    col("shd_Alarm").lower().id()
    .bounded_power(bound=0.0))

term("adj_precision_Alarm",
    col("adj_precision_Alarm").higher().id()
    .bounded_power(bound=1.0))

term("adj_recall_Alarm",
    col("adj_recall_Alarm").higher().id()
    .bounded_power(bound=1.0))

term("arrow_precision_Alarm",
    col("arrow_precision_Alarm").higher().id()
    .bounded_power(bound=1.0))

term("arrow_recall_Alarm",
    col("arrow_recall_Alarm").higher().id()
    .bounded_power(bound=1.0))

term("shd_Hailfinder",
    col("shd_Hailfinder").lower().id()
    .bounded_power(bound=0.0))

term("adj_precision_Hailfinder",
    col("adj_precision_Hailfinder").higher().id()
    .bounded_power(bound=1.0))

term("adj_recall_Hailfinder",
    col("adj_recall_Hailfinder").higher().id()
    .bounded_power(bound=1.0))

term("arrow_precision_Hailfinder",
    col("arrow_precision_Hailfinder").higher().id()
    .bounded_power(bound=1.0))

term("arrow_recall_Hailfinder",
    col("arrow_recall_Hailfinder").higher().id()
    .bounded_power(bound=1.0))

term("shd_Win95pts",
    col("shd_Win95pts").lower().id()
    .bounded_power(bound=0.0))

term("adj_precision_Win95pts",
    col("adj_precision_Win95pts").higher().id()
    .bounded_power(bound=1.0))

term("adj_recall_Win95pts",
    col("adj_recall_Win95pts").higher().id()
    .bounded_power(bound=1.0))

term("arrow_precision_Win95pts",
    col("arrow_precision_Win95pts").higher().id()
    .bounded_power(bound=1.0))

term("arrow_recall_Win95pts",
    col("arrow_recall_Win95pts").higher().id()
    .bounded_power(bound=1.0))

setting("Cancer", weighted_mean(("shd_Cancer", 1.0), ("adj_precision_Cancer", 1.0), ("adj_recall_Cancer", 1.0), ("arrow_precision_Cancer", 1.0), ("arrow_recall_Cancer", 1.0)))
setting("Child", weighted_mean(("shd_Child", 1.0), ("adj_precision_Child", 1.0), ("adj_recall_Child", 1.0), ("arrow_precision_Child", 1.0), ("arrow_recall_Child", 1.0)))
setting("Alarm", weighted_mean(("shd_Alarm", 1.0), ("adj_precision_Alarm", 1.0), ("adj_recall_Alarm", 1.0), ("arrow_precision_Alarm", 1.0), ("arrow_recall_Alarm", 1.0)))
setting("Hailfinder", weighted_mean(("shd_Hailfinder", 1.0), ("adj_precision_Hailfinder", 1.0), ("adj_recall_Hailfinder", 1.0), ("arrow_precision_Hailfinder", 1.0), ("arrow_recall_Hailfinder", 1.0)))
setting("Win95pts", weighted_mean(("shd_Win95pts", 1.0), ("adj_precision_Win95pts", 1.0), ("adj_recall_Win95pts", 1.0), ("arrow_precision_Win95pts", 1.0), ("arrow_recall_Win95pts", 1.0)))

task(gmean("Cancer", "Child", "Alarm", "Hailfinder", "Win95pts"))
