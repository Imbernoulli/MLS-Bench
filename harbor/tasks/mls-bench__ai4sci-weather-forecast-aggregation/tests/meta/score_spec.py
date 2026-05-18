"""Score spec for ai4sci-weather-forecast-aggregation.

Reference values are the mean across the four baseline aggregators
(cross_attention, mean_pooling, learned_weighted_sum, self_attention) on
their latest fresh leaderboard rows. Self_attention's current fresh row is
empty (run did not complete with the new fairness-fix workspace), so the
ref is the mean of the three completed baselines (cross_attention,
mean_pooling, learned_weighted_sum). Update this file once self_attention
re-runs successfully so the ref reflects all four methods.

Latest fresh values (is_final=false, post 2026-04-20):
  cross_attention:      z500=245.1297  t850=2.146   wind10m=3.2303
  mean_pooling:         z500=354.5569  t850=8.2828  wind10m=3.4344
  learned_weighted_sum: z500=349.5137  t850=4.6744  wind10m=3.4168
"""
from mlsbench.scoring.dsl import *

# Mean of {245.1297, 354.5569, 349.5137} = 316.4001
term("w_rmse_geopotential_500_z500_3day",
    col("w_rmse_geopotential_500_z500-3day").lower().id()
    .bounded_power(bound=0.0))

# Mean of {2.146, 8.2828, 4.6744} = 5.0344
term("w_rmse_temperature_850_t850_5day",
    col("w_rmse_temperature_850_t850-5day").lower().id()
    .bounded_power(bound=0.0))

# Mean of {3.2303, 3.4344, 3.4168} = 3.3605
term("w_rmse_10m_u_component_of_wind_wind10m_7day",
    col("w_rmse_10m_u_component_of_wind_wind10m-7day").lower().id()
    .bounded_power(bound=0.0))

setting("z500-3day", weighted_mean(("w_rmse_geopotential_500_z500_3day", 1.0)))
setting("t850-5day", weighted_mean(("w_rmse_temperature_850_t850_5day", 1.0)))
setting("wind10m-7day", weighted_mean(("w_rmse_10m_u_component_of_wind_wind10m_7day", 1.0)))

task(gmean("z500-3day", "t850-5day", "wind10m-7day"))
