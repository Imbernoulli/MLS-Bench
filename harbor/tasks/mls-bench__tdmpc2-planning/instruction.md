# MLS-Bench: tdmpc2-planning

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


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/tdmpc2/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `tdmpc2/tdmpc2/common/custom_planner.py`
- editable lines **15–120**


Other files you may **read** for context (do not modify):
- `tdmpc2/tdmpc2/tdmpc2.py`
- `tdmpc2/tdmpc2/common/world_model.py`
- `tdmpc2/tdmpc2/common/math.py`


## Readable Context


### `tdmpc2/tdmpc2/common/custom_planner.py`  [EDITABLE — lines 15–120 only]

```python
     1: """Custom planning algorithm for TD-MPC2.
     2: 
     3: Replace the planning logic in custom_plan() with your trajectory
     4: optimization method. The function is called at each environment step
     5: to select actions using the learned world model.
     6: """
     7: 
     8: import torch
     9: from common import math
    10: 
    11: 
    12: # =====================================================================
    13: # EDITABLE: Custom planning algorithm
    14: # =====================================================================
    15: @torch.no_grad()
    16: def custom_plan(agent, obs, t0=False, eval_mode=False, task=None):
    17:     """Plan a sequence of actions using the learned world model.
    18: 
    19:     Args:
    20:         agent: TDMPC2 agent instance with:
    21:             agent.cfg: configuration object with fields:
    22:                 horizon, num_samples, num_elites, num_pi_trajs,
    23:                 temperature, min_std, max_std, action_dim, multitask
    24:             agent.model: WorldModel with methods:
    25:                 encode(obs, task) -> z
    26:                 next(z, action, task) -> z'
    27:                 pi(z, task) -> (action, info_dict)
    28:                 Q(z, action, task, return_type='avg') -> Q-value
    29:                 reward(z, action, task) -> reward logits
    30:             agent._estimate_value(z, actions, task) -> value estimates
    31:             agent._prev_mean: Buffer(horizon, action_dim) for warm-start
    32:             agent.device: torch device
    33:         obs: observation tensor (1, obs_dim)
    34:         t0: True if first step in episode (resets warm-start)
    35:         eval_mode: True for deterministic action selection
    36:         task: task index (multi-task only, otherwise None)
    37: 
    38:     Returns:
    39:         action: selected action tensor (action_dim,), clamped to [-1, 1]
    40:     """
    41:     cfg = agent.cfg
    42: 
    43:     # Sample policy trajectories as warm-starts
    44:     z = agent.model.encode(obs, task)
    45:     if cfg.num_pi_trajs > 0:
    46:         pi_actions = torch.empty(
    47:             cfg.horizon, cfg.num_pi_trajs, cfg.action_dim,
    48:             device=agent.device,
    49:         )
    50:         _z = z.repeat(cfg.num_pi_trajs, 1)
    51:         for t in range(cfg.horizon - 1):
    52:             pi_actions[t], _ = agent.model.pi(_z, task)
    53:             _z = agent.model.next(_z, pi_actions[t], task)
    54:         pi_actions[-1], _ = agent.model.pi(_z, task)
    55: 
    56:     # Initialize sampling distribution
    57:     z = z.repeat(cfg.num_samples, 1)
    58:     mean = torch.zeros(cfg.horizon, cfg.action_dim, device=agent.device)
    59:     std = torch.full(
    60:         (cfg.horizon, cfg.action_dim), cfg.max_std,
    61:         dtype=torch.float, device=agent.device,
    62:     )
    63:     if not t0:
    64:         mean[:-1] = agent._prev_mean[1:]
    65:     actions = torch.empty(
    66:         cfg.horizon, cfg.num_samples, cfg.action_dim,
    67:         device=agent.device,
    68:     )
    69:     if cfg.num_pi_trajs > 0:
    70:         actions[:, :cfg.num_pi_trajs] = pi_actions
    71: 
    72:     # Iterate MPPI
    73:     for _ in range(cfg.iterations):
    74:         # Sample actions from Gaussian
    75:         r = torch.randn(
    76:             cfg.horizon, cfg.num_samples - cfg.num_pi_trajs,
    77:             cfg.action_dim, device=agent.device,
    78:         )
    79:         actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
    80:         actions_sample = actions_sample.clamp(-1, 1)
    81:         actions[:, cfg.num_pi_trajs:] = actions_sample
    82:         if cfg.multitask:
    83:             actions = actions * agent.model._action_masks[task]
    84: 
    85:         # Evaluate trajectories and select elites
    86:         value = agent._estimate_value(z, actions, task).nan_to_num(0)
    87:         elite_idxs = torch.topk(
    88:             value.squeeze(1), cfg.num_elites, dim=0,
    89:         ).indices
    90:         elite_value = value[elite_idxs]
    91:         elite_actions = actions[:, elite_idxs]
    92: 
    93:         # Update sampling distribution (softmax-weighted)
    94:         max_value = elite_value.max(0).values
    95:         score = torch.exp(cfg.temperature * (elite_value - max_value))
    96:         score = score / score.sum(0)
    97:         mean = (
    98:             (score.unsqueeze(0) * elite_actions).sum(dim=1)
    99:             / (score.sum(0) + 1e-9)
   100:         )
   101:         std = (
   102:             (
   103:                 score.unsqueeze(0)
   104:                 * (elite_actions - mean.unsqueeze(1)) ** 2
   105:             ).sum(dim=1)
   106:             / (score.sum(0) + 1e-9)
   107:         ).sqrt()
   108:         std = std.clamp(cfg.min_std, cfg.max_std)
   109:         if cfg.multitask:
   110:             mean = mean * agent.model._action_masks[task]
   111:             std = std * agent.model._action_masks[task]
   112: 
   113:     # Select action from elites via Gumbel sampling
   114:     rand_idx = math.gumbel_softmax_sample(score.squeeze(1))
   115:     actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
   116:     a, std_final = actions[0], std[0]
   117:     if not eval_mode:
   118:         a = a + std_final * torch.randn(cfg.action_dim, device=agent.device)
   119:     agent._prev_mean.copy_(mean)
   120:     return a.clamp(-1, 1)
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `cem` baseline — editable region  [READ-ONLY — reference implementation]

In `tdmpc2/tdmpc2/common/custom_planner.py`:

```python
Lines 15–81:
    12: # =====================================================================
    13: # EDITABLE: Custom planning algorithm
    14: # =====================================================================
    15: @torch.no_grad()
    16: def custom_plan(agent, obs, t0=False, eval_mode=False, task=None):
    17:     """CEM baseline -- Cross-Entropy Method for trajectory optimization."""
    18:     cfg = agent.cfg
    19: 
    20:     # Sample policy trajectories as warm-starts
    21:     z = agent.model.encode(obs, task)
    22:     if cfg.num_pi_trajs > 0:
    23:         pi_actions = torch.empty(
    24:             cfg.horizon, cfg.num_pi_trajs, cfg.action_dim,
    25:             device=agent.device,
    26:         )
    27:         _z = z.repeat(cfg.num_pi_trajs, 1)
    28:         for t in range(cfg.horizon - 1):
    29:             pi_actions[t], _ = agent.model.pi(_z, task)
    30:             _z = agent.model.next(_z, pi_actions[t], task)
    31:         pi_actions[-1], _ = agent.model.pi(_z, task)
    32: 
    33:     # Initialize sampling distribution
    34:     z = z.repeat(cfg.num_samples, 1)
    35:     mean = torch.zeros(cfg.horizon, cfg.action_dim, device=agent.device)
    36:     std = torch.full(
    37:         (cfg.horizon, cfg.action_dim), cfg.max_std,
    38:         dtype=torch.float, device=agent.device,
    39:     )
    40:     if not t0:
    41:         mean[:-1] = agent._prev_mean[1:]
    42:     actions = torch.empty(
    43:         cfg.horizon, cfg.num_samples, cfg.action_dim,
    44:         device=agent.device,
    45:     )
    46:     if cfg.num_pi_trajs > 0:
    47:         actions[:, :cfg.num_pi_trajs] = pi_actions
    48: 
    49:     # Iterate CEM
    50:     for _ in range(cfg.iterations):
    51:         # Sample actions from Gaussian
    52:         r = torch.randn(
    53:             cfg.horizon, cfg.num_samples - cfg.num_pi_trajs,
    54:             cfg.action_dim, device=agent.device,
    55:         )
    56:         actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
    57:         actions_sample = actions_sample.clamp(-1, 1)
    58:         actions[:, cfg.num_pi_trajs:] = actions_sample
    59:         if cfg.multitask:
    60:             actions = actions * agent.model._action_masks[task]
    61: 
    62:         # Evaluate trajectories and select elites
    63:         value = agent._estimate_value(z, actions, task).nan_to_num(0)
    64:         elite_idxs = torch.topk(
    65:             value.squeeze(1), cfg.num_elites, dim=0,
    66:         ).indices
    67:         elite_actions = actions[:, elite_idxs]
    68: 
    69:         # CEM update: simple mean/std of elites (no softmax weighting)
    70:         mean = elite_actions.mean(dim=1)
    71:         std = elite_actions.std(dim=1).clamp(cfg.min_std, cfg.max_std)
    72:         if cfg.multitask:
    73:             mean = mean * agent.model._action_masks[task]
    74:             std = std * agent.model._action_masks[task]
    75: 
    76:     # Select action: use the mean (deterministic) or sample
    77:     a = mean[0]
    78:     if not eval_mode:
    79:         a = a + std[0] * torch.randn(cfg.action_dim, device=agent.device)
    80:     agent._prev_mean.copy_(mean)
    81:     return a.clamp(-1, 1)
```

### `icem` baseline — editable region  [READ-ONLY — reference implementation]

In `tdmpc2/tdmpc2/common/custom_planner.py`:

```python
Lines 15–118:
    12: # =====================================================================
    13: # EDITABLE: Custom planning algorithm
    14: # =====================================================================
    15: @torch.no_grad()
    16: def custom_plan(agent, obs, t0=False, eval_mode=False, task=None):
    17:     """iCEM baseline -- improved CEM with colored noise and keep-elites."""
    18:     cfg = agent.cfg
    19:     colored_noise_beta = 0.5  # temporal correlation strength
    20:     noise_decay = 0.9  # per-iteration noise decay factor
    21:     keep_fraction = 0.1  # fraction of elites kept across iterations
    22: 
    23:     # Sample policy trajectories as warm-starts
    24:     z = agent.model.encode(obs, task)
    25:     if cfg.num_pi_trajs > 0:
    26:         pi_actions = torch.empty(
    27:             cfg.horizon, cfg.num_pi_trajs, cfg.action_dim,
    28:             device=agent.device,
    29:         )
    30:         _z = z.repeat(cfg.num_pi_trajs, 1)
    31:         for t in range(cfg.horizon - 1):
    32:             pi_actions[t], _ = agent.model.pi(_z, task)
    33:             _z = agent.model.next(_z, pi_actions[t], task)
    34:         pi_actions[-1], _ = agent.model.pi(_z, task)
    35: 
    36:     # Initialize sampling distribution
    37:     z = z.repeat(cfg.num_samples, 1)
    38:     mean = torch.zeros(cfg.horizon, cfg.action_dim, device=agent.device)
    39:     std = torch.full(
    40:         (cfg.horizon, cfg.action_dim), cfg.max_std,
    41:         dtype=torch.float, device=agent.device,
    42:     )
    43:     if not t0:
    44:         mean[:-1] = agent._prev_mean[1:]
    45:     actions = torch.empty(
    46:         cfg.horizon, cfg.num_samples, cfg.action_dim,
    47:         device=agent.device,
    48:     )
    49:     if cfg.num_pi_trajs > 0:
    50:         actions[:, :cfg.num_pi_trajs] = pi_actions
    51: 
    52:     n_keep = max(1, int(cfg.num_elites * keep_fraction))
    53:     kept_actions = None  # elites kept from previous iteration
    54:     noise_scale = 1.0
    55: 
    56:     # Iterate iCEM
    57:     for iteration in range(cfg.iterations):
    58:         n_new = cfg.num_samples - cfg.num_pi_trajs
    59:         if kept_actions is not None:
    60:             n_new = n_new - kept_actions.shape[1]
    61: 
    62:         # Generate temporally correlated (colored) noise
    63:         white_noise = torch.randn(
    64:             cfg.horizon, n_new, cfg.action_dim,
    65:             device=agent.device,
    66:         )
    67:         # Apply temporal smoothing: exponential moving average along horizon
    68:         colored_noise = torch.zeros_like(white_noise)
    69:         colored_noise[0] = white_noise[0]
    70:         for t in range(1, cfg.horizon):
    71:             colored_noise[t] = (
    72:                 colored_noise_beta * colored_noise[t - 1]
    73:                 + (1 - colored_noise_beta) * white_noise[t]
    74:             )
    75: 
    76:         # Sample actions
    77:         actions_sample = (
    78:             mean.unsqueeze(1)
    79:             + noise_scale * std.unsqueeze(1) * colored_noise
    80:         )
    81:         actions_sample = actions_sample.clamp(-1, 1)
    82: 
    83:         # Combine: policy trajs + kept elites + new samples
    84:         start_idx = cfg.num_pi_trajs
    85:         if kept_actions is not None:
    86:             actions[:, start_idx : start_idx + kept_actions.shape[1]] = kept_actions
    87:             start_idx += kept_actions.shape[1]
    88:         actions[:, start_idx : start_idx + n_new] = actions_sample
    89: 
    90:         if cfg.multitask:
    91:             actions = actions * agent.model._action_masks[task]
    92: 
    93:         # Evaluate trajectories and select elites
    94:         value = agent._estimate_value(z, actions, task).nan_to_num(0)
    95:         elite_idxs = torch.topk(
    96:             value.squeeze(1), cfg.num_elites, dim=0,
    97:         ).indices
    98:         elite_actions = actions[:, elite_idxs]
    99: 
   100:         # Keep top elites for next iteration
   101:         kept_actions = elite_actions[:, :n_keep]
   102: 
   103:         # Update distribution (simple CEM-style)
   104:         mean = elite_actions.mean(dim=1)
   105:         std = elite_actions.std(dim=1).clamp(cfg.min_std, cfg.max_std)
   106:         if cfg.multitask:
   107:             mean = mean * agent.model._action_masks[task]
   108:             std = std * agent.model._action_masks[task]
   109: 
   110:         # Decay noise for refinement
   111:         noise_scale *= noise_decay
   112: 
   113:     # Select action: use the mean
   114:     a = mean[0]
   115:     if not eval_mode:
   116:         a = a + std[0] * torch.randn(cfg.action_dim, device=agent.device)
   117:     agent._prev_mean.copy_(mean)
   118:     return a.clamp(-1, 1)
```

### `mppi` baseline — editable region  [READ-ONLY — reference implementation]

In `tdmpc2/tdmpc2/common/custom_planner.py`:

```python
Lines 15–97:
    12: # =====================================================================
    13: # EDITABLE: Custom planning algorithm
    14: # =====================================================================
    15: @torch.no_grad()
    16: def custom_plan(agent, obs, t0=False, eval_mode=False, task=None):
    17:     """MPPI baseline -- Model Predictive Path Integral control."""
    18:     cfg = agent.cfg
    19: 
    20:     # Sample policy trajectories as warm-starts
    21:     z = agent.model.encode(obs, task)
    22:     if cfg.num_pi_trajs > 0:
    23:         pi_actions = torch.empty(
    24:             cfg.horizon, cfg.num_pi_trajs, cfg.action_dim,
    25:             device=agent.device,
    26:         )
    27:         _z = z.repeat(cfg.num_pi_trajs, 1)
    28:         for t in range(cfg.horizon - 1):
    29:             pi_actions[t], _ = agent.model.pi(_z, task)
    30:             _z = agent.model.next(_z, pi_actions[t], task)
    31:         pi_actions[-1], _ = agent.model.pi(_z, task)
    32: 
    33:     # Initialize sampling distribution
    34:     z = z.repeat(cfg.num_samples, 1)
    35:     mean = torch.zeros(cfg.horizon, cfg.action_dim, device=agent.device)
    36:     std = torch.full(
    37:         (cfg.horizon, cfg.action_dim), cfg.max_std,
    38:         dtype=torch.float, device=agent.device,
    39:     )
    40:     if not t0:
    41:         mean[:-1] = agent._prev_mean[1:]
    42:     actions = torch.empty(
    43:         cfg.horizon, cfg.num_samples, cfg.action_dim,
    44:         device=agent.device,
    45:     )
    46:     if cfg.num_pi_trajs > 0:
    47:         actions[:, :cfg.num_pi_trajs] = pi_actions
    48: 
    49:     # Iterate MPPI
    50:     for _ in range(cfg.iterations):
    51:         # Sample actions from Gaussian
    52:         r = torch.randn(
    53:             cfg.horizon, cfg.num_samples - cfg.num_pi_trajs,
    54:             cfg.action_dim, device=agent.device,
    55:         )
    56:         actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
    57:         actions_sample = actions_sample.clamp(-1, 1)
    58:         actions[:, cfg.num_pi_trajs:] = actions_sample
    59:         if cfg.multitask:
    60:             actions = actions * agent.model._action_masks[task]
    61: 
    62:         # Evaluate trajectories and select elites
    63:         value = agent._estimate_value(z, actions, task).nan_to_num(0)
    64:         elite_idxs = torch.topk(
    65:             value.squeeze(1), cfg.num_elites, dim=0,
    66:         ).indices
    67:         elite_value = value[elite_idxs]
    68:         elite_actions = actions[:, elite_idxs]
    69: 
    70:         # Update sampling distribution (softmax-weighted)
    71:         max_value = elite_value.max(0).values
    72:         score = torch.exp(cfg.temperature * (elite_value - max_value))
    73:         score = score / score.sum(0)
    74:         mean = (
    75:             (score.unsqueeze(0) * elite_actions).sum(dim=1)
    76:             / (score.sum(0) + 1e-9)
    77:         )
    78:         std = (
    79:             (
    80:                 score.unsqueeze(0)
    81:                 * (elite_actions - mean.unsqueeze(1)) ** 2
    82:             ).sum(dim=1)
    83:             / (score.sum(0) + 1e-9)
    84:         ).sqrt()
    85:         std = std.clamp(cfg.min_std, cfg.max_std)
    86:         if cfg.multitask:
    87:             mean = mean * agent.model._action_masks[task]
    88:             std = std * agent.model._action_masks[task]
    89: 
    90:     # Select action from elites via Gumbel sampling
    91:     rand_idx = math.gumbel_softmax_sample(score.squeeze(1))
    92:     actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
    93:     a, std_final = actions[0], std[0]
    94:     if not eval_mode:
    95:         a = a + std_final * torch.randn(cfg.action_dim, device=agent.device)
    96:     agent._prev_mean.copy_(mean)
    97:     return a.clamp(-1, 1)
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
