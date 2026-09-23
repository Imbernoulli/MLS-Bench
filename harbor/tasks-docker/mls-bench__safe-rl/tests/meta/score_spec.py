"""Score spec for safe-rl.

Safe reinforcement learning task with three environments: SafetyPointGoal1-v0,
SafetyCarGoal1-v0, SafetyPointButton1-v0. Each environment produces two metrics:
  - episode_return (higher is better), unbounded
  - episode_cost (must be <= 25, the omnisafe / Safety-Gymnasium cost_limit)

Per environment the setting score is
    (0.9 * ret + 0.1 * feasible) * penalty(cost)

ret: sigmoid on the return. The floor is the worst baseline (DSL convention),
which here is pid_lag, a near-zero-return policy. The reference is not a
baseline. It is the published return of unconstrained PPO in OmniSafe's
on-policy benchmark (omnisafe benchmarks/on-policy/README.md, Table 2, "PPO"
column, cost_limit=25): PointGoal1 25.44, CarGoal1 33.41, PointButton1 27.03.
Matching unconstrained PPO's return scores 0.5. These agree with our own
naive (unconstrained PPO) baseline within seed noise (25.54 and 32.82;
PointButton1's 19.69 mean is dragged down by one weak seed, 8.79 vs
25.04 / 25.25). Fixing them keeps the return scale from moving with the noise
of a baseline that violates the constraint.

feasible: 1 when cost <= 25 (bounded_power with bound = the cost limit and
ref = the limit, so gamma = 1), falling linearly to 0 at the worst baseline's
cost. The worst baseline's return is the return floor, so without this term
the only constraint-satisfying baseline (pid_lag) scored exactly 0 and ranked
below the constraint-ignoring naive baseline. The 0.1 weight gives a feasible
but trivial policy 0.1, well below a feasible policy with real return (half
of unconstrained PPO's return at cost 20 scores about 0.31).

penalty(cost): exp(-0.15 * (cost - 25)) above the limit, 1 otherwise.
"""
from mlsbench.scoring.dsl import *

COST_LIMIT = 25.0
W_RET = 0.9
W_FEASIBLE = 0.1

# ---- SafetyPointGoal1-v0 ----
term("ret_point_goal",
    col("ep_ret_SafetyPointGoal1_v0")
    .higher().id()
    .sigmoid(ref=25.44)
)
term("feasible_point_goal",
    col("ep_cost_SafetyPointGoal1_v0")
    .lower().id()
    .bounded_power(bound=COST_LIMIT, ref=COST_LIMIT)
)
term("cost_point_goal",
    penalty_upper(col("ep_cost_SafetyPointGoal1_v0").lower().id(), target=COST_LIMIT, sharpness=0.15)
)

# ---- SafetyCarGoal1-v0 ----
term("ret_car_goal",
    col("ep_ret_SafetyCarGoal1_v0")
    .higher().id()
    .sigmoid(ref=33.41)
)
term("feasible_car_goal",
    col("ep_cost_SafetyCarGoal1_v0")
    .lower().id()
    .bounded_power(bound=COST_LIMIT, ref=COST_LIMIT)
)
term("cost_car_goal",
    penalty_upper(col("ep_cost_SafetyCarGoal1_v0").lower().id(), target=COST_LIMIT, sharpness=0.15)
)

# ---- SafetyPointButton1-v0 ----
term("ret_point_button",
    col("ep_ret_SafetyPointButton1_v0")
    .higher().id()
    .sigmoid(ref=27.03)
)
term("feasible_point_button",
    col("ep_cost_SafetyPointButton1_v0")
    .lower().id()
    .bounded_power(bound=COST_LIMIT, ref=COST_LIMIT)
)
term("cost_point_button",
    penalty_upper(col("ep_cost_SafetyPointButton1_v0").lower().id(), target=COST_LIMIT, sharpness=0.15)
)

# Settings (one per environment, with cost constraint)
setting("SafetyPointGoal1-v0",
    weighted_mean(("ret_point_goal", W_RET), ("feasible_point_goal", W_FEASIBLE)),
    constraints=["cost_point_goal"],
)
setting("SafetyCarGoal1-v0",
    weighted_mean(("ret_car_goal", W_RET), ("feasible_car_goal", W_FEASIBLE)),
    constraints=["cost_car_goal"],
)
setting("SafetyPointButton1-v0",
    weighted_mean(("ret_point_button", W_RET), ("feasible_point_button", W_FEASIBLE)),
    constraints=["cost_point_button"],
)

# Task: geometric mean across environments
task(gmean("SafetyPointGoal1-v0", "SafetyCarGoal1-v0", "SafetyPointButton1-v0"))
