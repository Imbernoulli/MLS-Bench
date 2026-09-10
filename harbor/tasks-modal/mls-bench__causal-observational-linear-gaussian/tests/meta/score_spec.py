"""Score spec for causal-observational-linear-gaussian."""
from mlsbench.scoring.dsl import *

# Config labels: ER10, ER20, SF50, SF50-Hard, ER20-Noisy
# Each label produces its own set of metrics via the parser using cmd_label as prefix.
# Metrics for ER10-Hard, ER20-Hard, ER10-Noisy, SF50-Noisy exist in leaderboard from
# historical runs but are not current config labels, so they are excluded.

term("shd_ER10",
    col("shd_ER10").lower().id()
    .bounded_power(bound=0.0))

term("adj_precision_ER10",
    col("adj_precision_ER10").higher().id()
    .bounded_power(bound=1.0))

term("adj_recall_ER10",
    col("adj_recall_ER10").higher().id()
    .bounded_power(bound=1.0))

term("arrow_precision_ER10",
    col("arrow_precision_ER10").higher().id()
    .bounded_power(bound=1.0))

term("arrow_recall_ER10",
    col("arrow_recall_ER10").higher().id()
    .bounded_power(bound=1.0))

term("shd_ER20",
    col("shd_ER20").lower().id()
    .bounded_power(bound=0.0))

term("adj_precision_ER20",
    col("adj_precision_ER20").higher().id()
    .bounded_power(bound=1.0))

term("adj_recall_ER20",
    col("adj_recall_ER20").higher().id()
    .bounded_power(bound=1.0))

term("arrow_precision_ER20",
    col("arrow_precision_ER20").higher().id()
    .bounded_power(bound=1.0))

term("arrow_recall_ER20",
    col("arrow_recall_ER20").higher().id()
    .bounded_power(bound=1.0))

term("shd_SF50",
    col("shd_SF50").lower().id()
    .bounded_power(bound=0.0))

term("adj_precision_SF50",
    col("adj_precision_SF50").higher().id()
    .bounded_power(bound=1.0))

term("adj_recall_SF50",
    col("adj_recall_SF50").higher().id()
    .bounded_power(bound=1.0))

term("arrow_precision_SF50",
    col("arrow_precision_SF50").higher().id()
    .bounded_power(bound=1.0))

term("arrow_recall_SF50",
    col("arrow_recall_SF50").higher().id()
    .bounded_power(bound=1.0))

term("shd_SF50_Hard",
    col("shd_SF50-Hard").lower().id()
    .bounded_power(bound=0.0))

term("adj_precision_SF50_Hard",
    col("adj_precision_SF50-Hard").higher().id()
    .bounded_power(bound=1.0))

term("adj_recall_SF50_Hard",
    col("adj_recall_SF50-Hard").higher().id()
    .bounded_power(bound=1.0))

term("arrow_precision_SF50_Hard",
    col("arrow_precision_SF50-Hard").higher().id()
    .bounded_power(bound=1.0))

term("arrow_recall_SF50_Hard",
    col("arrow_recall_SF50-Hard").higher().id()
    .bounded_power(bound=1.0))

term("shd_ER20_Noisy",
    col("shd_ER20-Noisy").lower().id()
    .bounded_power(bound=0.0))

term("adj_precision_ER20_Noisy",
    col("adj_precision_ER20-Noisy").higher().id()
    .bounded_power(bound=1.0))

term("adj_recall_ER20_Noisy",
    col("adj_recall_ER20-Noisy").higher().id()
    .bounded_power(bound=1.0))

term("arrow_precision_ER20_Noisy",
    col("arrow_precision_ER20-Noisy").higher().id()
    .bounded_power(bound=1.0))

term("arrow_recall_ER20_Noisy",
    col("arrow_recall_ER20-Noisy").higher().id()
    .bounded_power(bound=1.0))

setting("ER10", weighted_mean(("shd_ER10", 1.0), ("adj_precision_ER10", 1.0), ("adj_recall_ER10", 1.0), ("arrow_precision_ER10", 1.0), ("arrow_recall_ER10", 1.0)))
setting("ER20", weighted_mean(("shd_ER20", 1.0), ("adj_precision_ER20", 1.0), ("adj_recall_ER20", 1.0), ("arrow_precision_ER20", 1.0), ("arrow_recall_ER20", 1.0)))
setting("SF50", weighted_mean(("shd_SF50", 1.0), ("adj_precision_SF50", 1.0), ("adj_recall_SF50", 1.0), ("arrow_precision_SF50", 1.0), ("arrow_recall_SF50", 1.0)))
setting("SF50-Hard", weighted_mean(("shd_SF50_Hard", 1.0), ("adj_precision_SF50_Hard", 1.0), ("adj_recall_SF50_Hard", 1.0), ("arrow_precision_SF50_Hard", 1.0), ("arrow_recall_SF50_Hard", 1.0)))
setting("ER20-Noisy", weighted_mean(("shd_ER20_Noisy", 1.0), ("adj_precision_ER20_Noisy", 1.0), ("adj_recall_ER20_Noisy", 1.0), ("arrow_precision_ER20_Noisy", 1.0), ("arrow_recall_ER20_Noisy", 1.0)))

task(gmean("ER10", "ER20", "SF50", "SF50-Hard", "ER20-Noisy"))
