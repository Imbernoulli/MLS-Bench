"""Score spec for robomimic-obs-encoder — success_rate (higher is better).

Reference: task default baseline (simple concat observation encoder).
  ToolHang=0.15, Can=0.85, Square=0.73
"""
from mlsbench.scoring.dsl import *

term("success_rate_tool_hang_ph",
    col("success_rate_tool_hang_ph").higher().id()
    .bounded_power(bound=1.0))

term("success_rate_can_ph",
    col("success_rate_can_ph").higher().id()
    .bounded_power(bound=1.0))

term("success_rate_square_ph",
    col("success_rate_square_ph").higher().id()
    .bounded_power(bound=1.0))

setting("tool_hang_ph", weighted_mean(("success_rate_tool_hang_ph", 1.0)))
setting("can_ph", weighted_mean(("success_rate_can_ph", 1.0)))
setting("square_ph", weighted_mean(("success_rate_square_ph", 1.0)))

task(gmean("tool_hang_ph", "can_ph", "square_ph"))
