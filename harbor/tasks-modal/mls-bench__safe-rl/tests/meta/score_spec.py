"""Score spec for safe-rl.

Safe reinforcement learning task with three environments: SafetyPointGoal1-v0,
SafetyCarGoal1-v0, SafetyPointButton1-v0. Each environment produces two metrics:
  - episode_return (higher is better) — unbounded, use sigmoid
  - episode_cost (should be <= 25) — constraint penalty

The tension in safe RL is achieving high return while keeping cost under the
safety threshold. pid_lag achieves low cost but modest return; naive achieves
high return but violates the cost constraint heavily.

ref values for episode_return are set near the best baseline (naive, ~PPO at 2M
steps) so that best baseline scores ~0.5. Will recalibrate after baselines run.
The cost constraint target is 25.0 as specified by the safe RL formulation.
"""
from mlsbench.scoring.dsl import *

# ---- SafetyPointGoal1-v0 ----
term("ret_point_goal",
    col("ep_ret_SafetyPointGoal1_v0")
    .higher().id()
    .sigmoid()
)
term("cost_point_goal",
    penalty_upper(col("ep_cost_SafetyPointGoal1_v0").lower().id(), target=25.0, sharpness=0.15)
)

# ---- SafetyCarGoal1-v0 ----
term("ret_car_goal",
    col("ep_ret_SafetyCarGoal1_v0")
    .higher().id()
    .sigmoid()
)
term("cost_car_goal",
    penalty_upper(col("ep_cost_SafetyCarGoal1_v0").lower().id(), target=25.0, sharpness=0.15)
)

# ---- SafetyPointButton1-v0 ----
term("ret_point_button",
    col("ep_ret_SafetyPointButton1_v0")
    .higher().id()
    .sigmoid()
)
term("cost_point_button",
    penalty_upper(col("ep_cost_SafetyPointButton1_v0").lower().id(), target=25.0, sharpness=0.15)
)

# Settings (one per environment, with cost constraint)
setting("SafetyPointGoal1-v0",
    weighted_mean(("ret_point_goal", 1.0)),
    constraints=["cost_point_goal"],
)
setting("SafetyCarGoal1-v0",
    weighted_mean(("ret_car_goal", 1.0)),
    constraints=["cost_car_goal"],
)
setting("SafetyPointButton1-v0",
    weighted_mean(("ret_point_button", 1.0)),
    constraints=["cost_point_button"],
)

# Task: geometric mean across environments
task(gmean("SafetyPointGoal1-v0", "SafetyCarGoal1-v0", "SafetyPointButton1-v0"))
