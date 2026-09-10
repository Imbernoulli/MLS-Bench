"""Score spec for jepa-planning."""
from mlsbench.scoring.dsl import *

# success_rate: higher is better, bounded at 1.0
# mean_dist: distance from goal, lower is better, bounded below by 0.
# mean_steps_to_success: fewer steps to succeed, lower is better, bounded below by 0.
term("success_rate_horizon_30",
    col("success_rate_horizon-30").higher().id()
    .bounded_power(bound=1.0))

term("mean_dist_horizon_30",
    col("mean_dist_horizon-30").lower().id()
    .bounded_power(bound=0.0))

term("mean_steps_to_success_horizon_30",
    col("mean_steps_to_success_horizon-30").lower().id()
    .bounded_power(bound=0.0))

term("success_rate_horizon_60",
    col("success_rate_horizon-60").higher().id()
    .bounded_power(bound=1.0))

term("mean_dist_horizon_60",
    col("mean_dist_horizon-60").lower().id()
    .bounded_power(bound=0.0))

term("mean_steps_to_success_horizon_60",
    col("mean_steps_to_success_horizon-60").lower().id()
    .bounded_power(bound=0.0))

term("success_rate_horizon_90",
    col("success_rate_horizon-90").higher().id()
    .bounded_power(bound=1.0))

term("mean_dist_horizon_90",
    col("mean_dist_horizon-90").lower().id()
    .bounded_power(bound=0.0))

term("mean_steps_to_success_horizon_90",
    col("mean_steps_to_success_horizon-90").lower().id()
    .bounded_power(bound=0.0))

setting("horizon-30", weighted_mean(("success_rate_horizon_30", 1.0), ("mean_dist_horizon_30", 1.0), ("mean_steps_to_success_horizon_30", 1.0)))
setting("horizon-60", weighted_mean(("success_rate_horizon_60", 1.0), ("mean_dist_horizon_60", 1.0), ("mean_steps_to_success_horizon_60", 1.0)))
setting("horizon-90", weighted_mean(("success_rate_horizon_90", 1.0), ("mean_dist_horizon_90", 1.0), ("mean_steps_to_success_horizon_90", 1.0)))

task(gmean("horizon-30", "horizon-60", "horizon-90"))
