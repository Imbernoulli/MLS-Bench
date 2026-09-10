# Online RL: On-Policy Actor-Critic for Continuous Control

## Research Question
Design and implement an on-policy actor-critic RL algorithm for
continuous control. Your code goes in `custom_onpolicy_continuous.py`.
Several reference implementations are provided as read-only `*.edit.py`
baselines.

## Background
On-policy methods collect trajectories with the current policy, compute
advantages with Generalized Advantage Estimation (GAE), and update the
policy via mini-batch optimization on freshly collected data. Compared
to off-policy methods they avoid replay-distribution mismatch but are
less sample-efficient and more sensitive to update stability. Different
design points address these tensions through clipped surrogate
objectives, adaptive penalties, stochasticity injection,
advantage-weighted regression, or other policy-update rules.

Reference baselines spanning the design space:
- **PPO (clip)** — Schulman et al., "Proximal Policy Optimization
  Algorithms" (arXiv:1707.06347). Clipped surrogate with default clip
  range `epsilon = 0.2` and GAE `lambda = 0.95`.
- **PPO-Penalty** — KL-penalty variant from the same paper.
- **AWR** — Peng et al., "Advantage-Weighted Regression: Simple and
  Scalable Off-Policy Reinforcement Learning" (arXiv:1910.00177).
  Advantage-weighted supervised policy update with default temperature
  `beta = 1.0`.

## Constraints
- Network architecture dimensions are FIXED and cannot be modified.
- Total parameter count is enforced at runtime; the contribution must
  be algorithmic (action distribution, surrogate loss, penalty,
  exploration injection, value loss) rather than capacity.
- Do **not** simply copy a reference implementation with minor changes.

## Evaluation
Trained and evaluated on Gymnasium MuJoCo continuous-control
environments including HalfCheetah-v4, Hopper-v4 and Walker2d-v4 within
a fixed interaction budget. Metric: mean episodic return over
evaluation episodes (higher is better). Strong methods should remain
reliable across environments with different dynamics rather than tuning
to one.
