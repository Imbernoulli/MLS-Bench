"""Score spec for rl-onpolicy-continuous.

Three environments (settings), each with a single return metric (higher is better).
No theoretical bound → sigmoid normalization.
Normalization uses dynamic leaderboard anchors: worst baseline = 0-point floor,
best baseline = 50-point anchor.
"""
from mlsbench.scoring.dsl import *

# --- halfcheetah-v4 ---
term("return_halfcheetah",
    col("eval_return_halfcheetah_v4")
    .higher().id()
    .sigmoid()
)

# --- swimmer-v4 ---
term("return_swimmer",
    col("eval_return_swimmer_v4")
    .higher().id()
    .sigmoid()
)

# --- inverteddoublependulum-v4 ---
term("return_invpendulum",
    col("eval_return_inverteddoublependulum_v4")
    .higher().id()
    .sigmoid()
)

# Settings (one per environment)
setting("halfcheetah-v4", weighted_mean(("return_halfcheetah", 1.0)))
setting("swimmer-v4", weighted_mean(("return_swimmer", 1.0)))
setting("inverteddoublependulum-v4", weighted_mean(("return_invpendulum", 1.0)))

# Task: geometric mean across environments
task(gmean("halfcheetah-v4", "swimmer-v4", "inverteddoublependulum-v4"))
