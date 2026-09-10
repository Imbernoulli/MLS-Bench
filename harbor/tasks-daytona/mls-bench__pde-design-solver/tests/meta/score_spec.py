"""Score spec for pde-design-solver."""
from mlsbench.scoring.dsl import *

# rho_d: Spearman drag-correlation metric parsed as "drag correlation" -> higher better,
# bounded above by 1.0.
# c_d: drag coefficient -> lower is better for aerodynamic design
# l2_press, l2_velo: L2 errors -> lower better, bounded at 0

term("rho_d_Car",
    col("rho_d_Car").higher().id()
    .bounded_power(bound=1.0))

term("c_d_Car",
    col("c_d_Car").lower().id()
    .bounded_power(bound=0.0))

term("l2_press_Car",
    col("l2_press_Car").lower().id()
    .bounded_power(bound=0.0))

term("l2_velo_Car",
    col("l2_velo_Car").lower().id()
    .bounded_power(bound=0.0))

term("l2_press_AirfRANS",
    col("l2_press_AirfRANS").lower().id()
    .bounded_power(bound=0.0))

term("l2_velo_AirfRANS",
    col("l2_velo_AirfRANS").lower().id()
    .bounded_power(bound=0.0))

term("l2_press_AirCraft",
    col("l2_press_AirCraft").lower().id()
    .bounded_power(bound=0.0))

term("l2_velo_AirCraft",
    col("l2_velo_AirCraft").lower().id()
    .bounded_power(bound=0.0))

setting("Car", weighted_mean(("rho_d_Car", 1.0), ("c_d_Car", 1.0), ("l2_press_Car", 1.0), ("l2_velo_Car", 1.0)))
setting("AirfRANS", weighted_mean(("l2_press_AirfRANS", 1.0), ("l2_velo_AirfRANS", 1.0)))
setting("AirCraft", weighted_mean(("l2_press_AirCraft", 1.0), ("l2_velo_AirCraft", 1.0)))

task(gmean("Car", "AirfRANS", "AirCraft"))
