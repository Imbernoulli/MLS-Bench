"""Score spec for ai4sci-climate-emulation.

Normalization uses dynamic leaderboard anchors: the worst baseline is the
0-point floor and the best baseline is the 50-point anchor. The previous
auto-generated refs were inflated by accidentally pulling std-rows; refs are
no longer hand-coded here.
"""

from mlsbench.scoring.dsl import *

# ============================================================================
# short-30ep
# ============================================================================
term("nmse_short",
    col("nmse_short-30ep").lower().id()
    .bounded_power(bound=0.0))

term("rmse_short",
    col("rmse_short-30ep").lower().id()
    .bounded_power(bound=0.0))

term("ml_nmse_short",
    col("ml_nmse_short-30ep").lower().id()
    .bounded_power(bound=0.0))

term("sl_nmse_short",
    col("sl_nmse_short-30ep").lower().id()
    .bounded_power(bound=0.0))

term("r2_short",
    col("r2_short-30ep").higher().id()
    .bounded_power(bound=1.0))

# ============================================================================
# medium-100ep
# ============================================================================
term("nmse_medium",
    col("nmse_medium-100ep").lower().id()
    .bounded_power(bound=0.0))

term("rmse_medium",
    col("rmse_medium-100ep").lower().id()
    .bounded_power(bound=0.0))

term("ml_nmse_medium",
    col("ml_nmse_medium-100ep").lower().id()
    .bounded_power(bound=0.0))

term("sl_nmse_medium",
    col("sl_nmse_medium-100ep").lower().id()
    .bounded_power(bound=0.0))

term("r2_medium",
    col("r2_medium-100ep").higher().id()
    .bounded_power(bound=1.0))

# ============================================================================
# long-200ep (hidden test env)
# ============================================================================
term("nmse_long",
    col("nmse_long-200ep").lower().id()
    .bounded_power(bound=0.0))

term("rmse_long",
    col("rmse_long-200ep").lower().id()
    .bounded_power(bound=0.0))

term("ml_nmse_long",
    col("ml_nmse_long-200ep").lower().id()
    .bounded_power(bound=0.0))

term("sl_nmse_long",
    col("sl_nmse_long-200ep").lower().id()
    .bounded_power(bound=0.0))

term("r2_long",
    col("r2_long-200ep").higher().id()
    .bounded_power(bound=1.0))

# ============================================================================
# Per-setting & overall score
# ============================================================================
setting("short-30ep", weighted_mean(
    ("nmse_short", 1.0), ("rmse_short", 1.0),
    ("ml_nmse_short", 1.0), ("sl_nmse_short", 1.0),
    ("r2_short", 1.0)))

setting("medium-100ep", weighted_mean(
    ("nmse_medium", 1.0), ("rmse_medium", 1.0),
    ("ml_nmse_medium", 1.0), ("sl_nmse_medium", 1.0),
    ("r2_medium", 1.0)))

setting("long-200ep", weighted_mean(
    ("nmse_long", 1.0), ("rmse_long", 1.0),
    ("ml_nmse_long", 1.0), ("sl_nmse_long", 1.0),
    ("r2_long", 1.0)))

task(gmean("short-30ep", "medium-100ep", "long-200ep"))
