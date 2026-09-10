# JEPA World Model Planning: Algorithm Design

## Objective
Design a planning algorithm that exploits a learned JEPA (Joint Embedding Predictive Architecture) world model for goal-conditioned navigation. The evaluation uses a Two Rooms environment in which the agent must navigate around walls and through doorways to reach a randomly sampled goal location.

## Research Question
Can you design a planning algorithm that outperforms standard derivative-free methods such as CEM and MPPI by better exploiting the structure of a learned JEPA world model?

## Background
JEPA world models predict future latent encodings rather than future observations, so planning happens entirely in representation space. Recent work studies offline learning of latent dynamics models with JEPA-style objectives and uses derivative-free planners on top of them — see Sobal et al. 2025, "Learning from Reward-Free Offline Data: A Case for Planning with Latent Dynamics Models" (arXiv:2502.14819). Two standard derivative-free planners used for this kind of latent world model are:
- **CEM (Cross-Entropy Method)**: iteratively refits a Gaussian over action sequences using the top-`k` elites under the model rollout cost.
- **MPPI (Model Predictive Path Integral)**: importance-weighted update over sampled action sequences (Williams et al., arXiv:1509.01149).

The JEPA world model checkpoint is fixed and provided by the evaluation environment; the task is to improve planning, not to retrain the model.

## What You Can Modify
You implement the `CustomPlanner` class in `custom_planner.py`. The class extends the `Planner` abstract base class and must implement the `plan()` method.

## Interface

### CustomPlanner Constructor
```python
def __init__(self, unroll, action_dim=2, plan_length=15,
             num_samples=200, n_iters=20, **kwargs):
```
- `unroll`: function that forward-simulates through the world model
- `action_dim`: action space dimensionality (2 for x/y movement)
- `plan_length`: maximum planning horizon
- `num_samples`: number of action samples (adjustable)
- `n_iters`: number of optimization iterations (adjustable)

### plan() Method
```python
def plan(self, obs_init, steps_left=None, eval_mode=True,
         t0=False, plan_vis_path=None) -> PlanningResult:
```
- `obs_init`: initial observation encoding `[1, C, 1, H, W]`
- `steps_left`: remaining steps in the episode
- Returns: `PlanningResult(actions=Tensor[T, A], ...)`

### Available Methods (Inherited)
- `self.unroll(obs_init, actions)`: forward-simulate actions through the world model.
  - `obs_init`: `[1, C, 1, H, W]` initial observation encoding
  - `actions`: `[B, A, T]` batch of action sequences
  - Returns: `[B, D, T+1, H, W]` predicted state encodings
- `self.objective(encodings)`: compute cost for predicted state encodings.
  - `encodings`: `[B, D, T, H, W]`
  - Returns: `[B]` cost per sample (lower is better)
- `self.cost_function(actions, obs_init)`: convenience method that calls `unroll` then `objective`.
  - Returns: `[B]` cost per sample

## Evaluation
- Environment: Two Rooms (65×65 grid with wall and door)
- Episodes: 20 with random start and goal positions controlled by the MLS-Bench SEED
- Max steps per episode: 200
- Success threshold: Euclidean distance < 4.5 from goal
- Benchmarks: three planning horizons (30, 60, 90 steps) test the algorithm across short, medium, and long-range planning
- Metric: `success_rate` (fraction of successful episodes) per horizon, higher is better
