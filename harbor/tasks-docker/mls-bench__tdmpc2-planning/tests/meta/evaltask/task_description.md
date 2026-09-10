# Planning Algorithm for Model-Based RL

## Objective
Design and implement a custom trajectory optimization algorithm for online planning in model-based reinforcement learning. Your code goes in the `custom_plan()` function in `custom_planner.py`. This function is called at every environment step to select actions using the learned world model.

## Background
**TD-MPC2** (Hansen, Su, Wang, ICLR 2024, arXiv:2310.16828) — the scalable successor of TD-MPC (Hansen, Wang, Su, ICML 2022, arXiv:2203.04955) — uses **Model Predictive Path Integral (MPPI)** (Williams et al., arXiv:1509.01149) for planning. At each step, the agent:
1. Samples `num_pi_trajs = 24` trajectories from the learned policy as warm-starts.
2. Iterates `iterations = 6` rounds of:
   - Sample `num_samples = 512` action sequences from N(mean, std).
   - Roll out each trajectory through the latent dynamics model for `horizon = 3` steps.
   - Estimate trajectory value using predicted rewards + terminal Q-value.
   - Select `num_elites = 64` best trajectories.
   - Update mean / std using softmax-weighted (temperature = 0.5) elite statistics.
3. Selects the final action via Gumbel-softmax sampling from elites.

Alternative planning approaches could improve sample efficiency, convergence speed, or final performance:
- **Cross-Entropy Method (CEM)**: simpler elite selection without softmax weighting.
- **iCEM** (Pinneri et al., CoRL 2020, arXiv:2008.06389): improved CEM with temporally correlated (colored) noise and keep-elites.
- **Gradient-based planning**: backpropagating through the world model.
- **Hybrid approaches**: combining sampling with gradient refinement.
- **Adaptive methods**: adjusting sampling parameters during optimization.

## What You Can Modify
The `custom_plan()` function in `custom_planner.py`. You have access to:
- `agent.model`: WorldModel with `encode`, `next`, `pi`, `Q`, `reward` methods
- `agent._estimate_value(z, actions, task)`: evaluates trajectory returns
- `agent._prev_mean`: warm-start buffer from previous planning step
- `agent.cfg`: all configuration parameters (horizon, num_samples, etc.)
- `common.math`: utility functions (`gumbel_softmax_sample`, `two_hot_inv`, etc.)

## Evaluation
- **Metric**: episode reward (higher is better)
- **Environments**: DMControl walker-walk and cheetah-run
- **Model**: TD-MPC2 with 1M parameters, 200K training steps
- **Note**: the planning algorithm affects both data collection quality during training and action selection during evaluation.

## Key Constraints
- The function must return a single action tensor of shape `(action_dim,)` clamped to `[-1, 1]`.
- The function runs under `@torch.no_grad()` — no gradient computation.
- Must update `agent._prev_mean` for temporal consistency across steps.
- Planning budget: keep total computation comparable to the default (6 iterations × 512 samples).
