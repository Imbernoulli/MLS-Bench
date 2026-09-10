# Safe RL: Constraint-Handling Mechanism Design

## Research Question
Design a constraint-handling mechanism for safe reinforcement learning.
Your code goes in `custom_lag.py`, a subclass of PPO registered as
`CustomLag`. Reference implementations using a Lagrange multiplier
(PPOLag) and a PID controller (CPPOPID) are provided as read-only
`*.edit.py` baselines.

## Background
Safe RL aims to maximize reward while keeping a long-run cost (e.g.
the count of safety violations) below a fixed limit. The standard
approach formulates the problem as a constrained MDP and converts it to
an unconstrained dual problem via a multiplier `lambda` updated from
the running cost violation. The mechanism that updates this multiplier
and combines reward and cost advantages directly determines the
agent's safety behavior:

- **naive** — constraint-unaware PPO baseline that ignores the safety
  constraint entirely; provides an upper bound on reward with no cost
  control.
- **PPOLag** — the multiplier is treated as a learnable parameter
  optimized by Adam to satisfy the dual objective. Simple but slow to
  react and prone to oscillation.
- **CPPOPID** — Stooke, Achiam and Abbeel, "Responsive Safety in
  Reinforcement Learning by PID Lagrangian Methods"
  (arXiv:2007.03964, ICML 2020). Replaces the integral-only Lagrange
  update with a PID controller; the benchmark uses the paper-style
  CPPOPID configuration with gains `kp = 0.1`, `ki = 0.01`,
  `kd = 0.01` and a derivative delay window of 10 epochs (matching
  `omnisafe/common/pid_lagrange.py`).

You must design:
1. A multiplier update rule in `_update()`.
2. An advantage combination formula in `_compute_adv_surrogate()` that
   blends the reward advantage `adv_r` and cost advantage `adv_c` using
   the current multiplier (e.g. `(adv_r - lam * adv_c) / (1 + lam)` in
   the standard Lagrangian baseline).

The PPO rollout loop, value functions, optimizer, environment
interface, and registration plumbing are fixed.

## Evaluation
Evaluated on Safety-Gymnasium navigation environments including:
- **SafetyPointGoal1-v0** — point robot navigating to goals while
  avoiding hazards.
- **SafetyCarGoal1-v0** — non-holonomic car robot with the same goal
  structure.
- **SafetyPointButton1-v0** — point robot pressing goal buttons while
  avoiding hazards.

Each environment trains for the benchmark's fixed step budget.
Metrics:
- Episode return (`reward`) — higher is better.
- Episode cost (`cost`) — lower is better, with a target threshold of
  25.0 per the Safety-Gymnasium convention used in `omnisafe`.

A method should achieve high return only when the cost constraint is
controlled across all environments.
