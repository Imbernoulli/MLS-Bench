# MLS-Bench: safe-rl

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

The agent is evaluated on safe-navigation environments where a robot must reach goals while avoiding hazards. A method should achieve high return while keeping the cost (safety violation count) below the configured cost limit across all environments.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/omnisafe/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits that change code outside these ranges — or creating new files, or
deleting whole files — will cause your submission to be invalid.

The line numbers mark an editable **region**, not a fixed line-count budget: you
may add or remove lines inside it. Only code outside the editable ranges must
stay unchanged.

- `omnisafe/omnisafe/algorithms/on_policy/naive_lagrange/custom_lag.py`
- editable lines **20–20**
- editable lines **48–70**


Other files you may **read** for context (do not modify):
- `omnisafe/omnisafe/common/lagrange.py`
- `omnisafe/omnisafe/common/pid_lagrange.py`
- `omnisafe/omnisafe/algorithms/on_policy/base/ppo.py`


## Readable Context


### `omnisafe/omnisafe/algorithms/on_policy/naive_lagrange/custom_lag.py`  [EDITABLE — lines 20–20, lines 48–70 only]

```python
     1: """Custom Lagrangian-based safe PPO for MLS-Bench.
     2: 
     3: EDITABLE section: imports + constraint handling methods.
     4: FIXED sections: algorithm registration, learn() with metrics reporting.
     5: """
     6: 
     7: from __future__ import annotations
     8: 
     9: import time
    10: 
    11: import numpy as np
    12: import torch
    13: 
    14: from omnisafe.algorithms import registry
    15: from omnisafe.algorithms.on_policy.base.ppo import PPO
    16: 
    17: # ===================================================================
    18: # EDITABLE: Custom imports
    19: # ===================================================================
    20: 
    21: 
    22: # ===================================================================
    23: # FIXED: Algorithm class definition
    24: # ===================================================================
    25: @registry.register
    26: class CustomLag(PPO):
    27:     """Custom Lagrangian-based safe RL algorithm.
    28: 
    29:     Extends PPO with constraint handling for safe reinforcement learning.
    30:     The agent must design:
    31:       1. _init: Initialize constraint handler state (call super()._init() first)
    32:       2. _init_log: Register logging keys (call super()._init_log() first)
    33:       3. _update: Update lagrangian multiplier, then call super()._update()
    34:       4. _compute_adv_surrogate: Combine reward and cost advantages
    35: 
    36:     Available config:
    37:         self._cfgs.lagrange_cfgs.cost_limit   (float, default 25.0)
    38:         self._cfgs.lagrange_cfgs.lambda_lr    (float, default 0.035)
    39: 
    40:     Available logger:
    41:         self._logger.get_stats('Metrics/EpCost')[0]  -- current mean episode cost
    42:         self._logger.store({'key': value})             -- log a metric value
    43:     """
    44: 
    45:     # ===============================================================
    46:     # EDITABLE: Constraint handling mechanism
    47:     # ===============================================================
    48:     def _init(self) -> None:
    49:         super()._init()
    50:         self._cost_limit: float = self._cfgs.lagrange_cfgs.cost_limit
    51:         self._lagrangian_multiplier: float = 0.0
    52: 
    53:     def _init_log(self) -> None:
    54:         super()._init_log()
    55:         self._logger.register_key('Metrics/LagrangeMultiplier', min_and_max=True)
    56: 
    57:     def _update(self) -> None:
    58:         Jc = self._logger.get_stats('Metrics/EpCost')[0]
    59:         assert not np.isnan(Jc), 'cost is nan'
    60:         # Default: no multiplier update -- agent should design this
    61:         super()._update()
    62:         self._logger.store({'Metrics/LagrangeMultiplier': self._lagrangian_multiplier})
    63: 
    64:     def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
    65:         """Combine reward and cost advantages.
    66: 
    67:         Default: only use reward advantage (ignores safety constraints entirely).
    68:         Agent should incorporate self._lagrangian_multiplier to penalize cost.
    69:         """
    70:         return adv_r
    71: 
    72:     # ===============================================================
    73:     # FIXED: Training loop with MLS-Bench metrics reporting
    74:     # ===============================================================
    75:     def learn(self) -> tuple[float, float, float]:
    76:         """Training loop with TRAIN_METRICS and TEST_METRICS output.
    77: 
    78:         The reported ep_ret / ep_cost / ep_len are the mean of the last 100
    79:         episodes (OmniSafe's default ``logger_cfgs.window_lens``), taken from a
    80:         copy that only the rollout below writes to: values stored into
    81:         ``self._logger`` elsewhere do not reach them.
    82: 
    83:         The environment steps the rollout takes are counted against the
    84:         ``--total-steps`` budget on the command line (not ``self._cfgs``,
    85:         which editable code can change). A run that is over its pro-rata share
    86:         of the budget at the end of any epoch is aborted.
    87:         """
    88:         import argparse
    89:         import os
    90:         from collections import deque
    91: 
    92:         from omnisafe.utils.distributed import dist_statistics_scalar
    93: 
    94:         start_time = time.time()
    95:         self._logger.log('INFO: Start training')
    96: 
    97:         cli = argparse.ArgumentParser(add_help=False)
    98:         cli.add_argument('--total-steps', type=int)
    99:         budget = cli.parse_known_args()[0].total_steps
   100:         if budget is None:  # not launched through train_safe_rl.py
   101:             budget = int(self._cfgs.train_cfgs.total_steps)
   102:         epochs = self._cfgs.train_cfgs.epochs
   103:         env_steps = 0
   104: 
   105:         logger = self._logger
   106:         episodes = {key: deque(maxlen=100) for key in ('Metrics/EpRet', 'Metrics/EpCost', 'Metrics/EpLen')}
   107: 
   108:         class RolloutLogger:
   109:             """Forwards to the logger and keeps a copy of the episode statistics."""
   110: 
   111:             def __getattr__(self, name):
   112:                 return getattr(logger, name)
   113: 
   114:             def store(self, data=None, /, **kwargs):
   115:                 nonlocal env_steps
   116:                 logger.store(data, **kwargs)
   117:                 if data is not None:
   118:                     kwargs.update(data)
   119:                 if 'Value/reward' in kwargs:  # stored once per rollout step, one value per env
   120:                     env_steps += kwargs['Value/reward'].numel()
   121:                 for key in episodes.keys() & kwargs.keys():
   122:                     val = kwargs[key]  # converted as Logger.store converts it
   123:                     if isinstance(val, torch.Tensor):
   124:                         val = val.mean().item()
   125:                     elif isinstance(val, np.ndarray):
   126:                         val = val.mean()
   127:                     episodes[key].append(val)
   128: 
   129:         def episode_mean(key: str) -> float:
   130:             """What ``Logger.get_stats(key)[0]`` computes, over the copy."""
   131:             vals = torch.tensor(list(episodes[key])).to(os.getenv('OMNISAFE_DEVICE', 'cpu'))
   132:             return dist_statistics_scalar(vals)[0].item()
   133: 
   134:         for epoch in range(epochs):
   135:             epoch_time = time.time()
   136: 
   137:             rollout_time = time.time()
   138:             self._env.rollout(
   139:                 steps_per_epoch=self._steps_per_epoch,
   140:                 agent=self._actor_critic,
   141:                 buffer=self._buf,
   142:                 logger=RolloutLogger(),
   143:             )
   144:             if env_steps > budget * (epoch + 1) // epochs:
   145:                 raise RuntimeError(
   146:                     f'MLS-Bench step budget exceeded: {env_steps} environment steps after '
   147:                     f'{epoch + 1} of {epochs} epochs, over the pro-rata share of '
   148:                     f'--total-steps {budget}',
   149:                 )
   150:             self._logger.store({'Time/Rollout': time.time() - rollout_time})
   151: 
   152:             update_time = time.time()
   153:             self._update()
   154:             self._logger.store({'Time/Update': time.time() - update_time})
   155: 
   156:             if self._cfgs.model_cfgs.exploration_noise_anneal:
   157:                 self._actor_critic.annealing(epoch)
   158: 
   159:             if self._cfgs.model_cfgs.actor.lr is not None:
   160:                 self._actor_critic.actor_scheduler.step()
   161: 
   162:             self._logger.store(
   163:                 {
   164:                     'TotalEnvSteps': (epoch + 1) * self._cfgs.algo_cfgs.steps_per_epoch,
   165:                     'Time/FPS': self._cfgs.algo_cfgs.steps_per_epoch / (time.time() - epoch_time),
   166:                     'Time/Total': (time.time() - start_time),
   167:                     'Time/Epoch': (time.time() - epoch_time),
   168:                     'Train/Epoch': epoch,
   169:                     'Train/LR': (
   170:                         0.0
   171:                         if self._cfgs.model_cfgs.actor.lr is None
   172:                         else self._actor_critic.actor_scheduler.get_last_lr()[0]
   173:                     ),
   174:                 },
   175:             )
   176: 
   177:             self._logger.dump_tabular()
   178: 
   179:             # -- MLS-Bench: TRAIN_METRICS --
   180:             _ep_ret = episode_mean('Metrics/EpRet')
   181:             _ep_cost = episode_mean('Metrics/EpCost')
   182:             _ep_len = episode_mean('Metrics/EpLen')
   183:             print(
   184:                 f'TRAIN_METRICS epoch={epoch} '
   185:                 f'ep_ret={_ep_ret:.4f} ep_cost={_ep_cost:.4f} '
   186:                 f'ep_len={_ep_len:.1f}',
   187:                 flush=True,
   188:             )
   189: 
   190:             if (epoch + 1) % self._cfgs.logger_cfgs.save_model_freq == 0 or (
   191:                 epoch + 1
   192:             ) == self._cfgs.train_cfgs.epochs:
   193:                 self._logger.torch_save()
   194: 
   195:         ep_ret = episode_mean('Metrics/EpRet')
   196:         ep_cost = episode_mean('Metrics/EpCost')
   197:         ep_len = episode_mean('Metrics/EpLen')
   198: 
   199:         # -- MLS-Bench: TEST_METRICS --
   200:         print(
   201:             f'TEST_METRICS ep_ret={ep_ret:.4f} ep_cost={ep_cost:.4f} '
   202:             f'ep_len={ep_len:.1f}',
   203:             flush=True,
   204:         )
   205: 
   206:         self._logger.close()
   207:         self._env.close()
   208: 
   209:         return ep_ret, ep_cost, ep_len
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `naive` baseline — editable region  [READ-ONLY — reference implementation]

In `omnisafe/omnisafe/algorithms/on_policy/naive_lagrange/custom_lag.py`:

```python
Lines 20–20:
    17: # ===================================================================
    18: # EDITABLE: Custom imports
    19: # ===================================================================
    20: 
    21: 
    22: # ===================================================================
    23: # FIXED: Algorithm class definition

Lines 48–65:
    45:     # ===============================================================
    46:     # EDITABLE: Constraint handling mechanism
    47:     # ===============================================================
    48:     def _init(self) -> None:
    49:         super()._init()
    50:         self._lagrangian_multiplier: float = 0.0
    51: 
    52:     def _init_log(self) -> None:
    53:         super()._init_log()
    54:         self._logger.register_key('Metrics/LagrangeMultiplier', min_and_max=True)
    55: 
    56:     def _update(self) -> None:
    57:         Jc = self._logger.get_stats('Metrics/EpCost')[0]
    58:         assert not np.isnan(Jc), 'cost is nan'
    59:         # Naive: no multiplier update, stays at 0
    60:         super()._update()
    61:         self._logger.store({'Metrics/LagrangeMultiplier': self._lagrangian_multiplier})
    62: 
    63:     def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
    64:         """Naive: ignore cost advantage entirely, optimize reward only."""
    65:         return adv_r
    66: 
    67:     # ===============================================================
    68:     # FIXED: Training loop with MLS-Bench metrics reporting
```

### `ppo_lag` baseline — editable region  [READ-ONLY — reference implementation]

In `omnisafe/omnisafe/algorithms/on_policy/naive_lagrange/custom_lag.py`:

```python
Lines 20–20:
    17: # ===================================================================
    18: # EDITABLE: Custom imports
    19: # ===================================================================
    20: 
    21: 
    22: # ===================================================================
    23: # FIXED: Algorithm class definition

Lines 48–79:
    45:     # ===============================================================
    46:     # EDITABLE: Constraint handling mechanism
    47:     # ===============================================================
    48:     def _init(self) -> None:
    49:         super()._init()
    50:         self._cost_limit: float = self._cfgs.lagrange_cfgs.cost_limit
    51:         init_value = max(self._cfgs.lagrange_cfgs.lagrangian_multiplier_init, 0.0)
    52:         self._lagrangian_multiplier = torch.nn.Parameter(
    53:             torch.as_tensor(init_value), requires_grad=True,
    54:         )
    55:         self._lambda_optimizer = torch.optim.Adam(
    56:             [self._lagrangian_multiplier],
    57:             lr=self._cfgs.lagrange_cfgs.lambda_lr,
    58:         )
    59: 
    60:     def _init_log(self) -> None:
    61:         super()._init_log()
    62:         self._logger.register_key('Metrics/LagrangeMultiplier', min_and_max=True)
    63: 
    64:     def _update(self) -> None:
    65:         Jc = self._logger.get_stats('Metrics/EpCost')[0]
    66:         assert not np.isnan(Jc), 'cost is nan'
    67:         # Lagrange multiplier update via Adam
    68:         self._lambda_optimizer.zero_grad()
    69:         lambda_loss = -self._lagrangian_multiplier * (Jc - self._cost_limit)
    70:         lambda_loss.backward()
    71:         self._lambda_optimizer.step()
    72:         self._lagrangian_multiplier.data.clamp_(0.0)
    73:         super()._update()
    74:         self._logger.store({'Metrics/LagrangeMultiplier': self._lagrangian_multiplier.item()})
    75: 
    76:     def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
    77:         """PPOLag: penalize cost advantage using Lagrange multiplier."""
    78:         penalty = self._lagrangian_multiplier.item()
    79:         return (adv_r - penalty * adv_c) / (1 + penalty)
    80: 
    81:     # ===============================================================
    82:     # FIXED: Training loop with MLS-Bench metrics reporting
```

### `pid_lag` baseline — editable region  [READ-ONLY — reference implementation]

In `omnisafe/omnisafe/algorithms/on_policy/naive_lagrange/custom_lag.py`:

```python
Lines 20–20:
    17: # ===================================================================
    18: # EDITABLE: Custom imports
    19: # ===================================================================
    20: from collections import deque
    21: 
    22: # ===================================================================
    23: # FIXED: Algorithm class definition

Lines 48–85:
    45:     # ===============================================================
    46:     # EDITABLE: Constraint handling mechanism
    47:     # ===============================================================
    48:     def _init(self) -> None:
    49:         super()._init()
    50:         self._cost_limit: float = self._cfgs.lagrange_cfgs.cost_limit
    51:         # PID controller gains (CPPOPID defaults)
    52:         self._pid_kp: float = 0.1
    53:         self._pid_ki: float = 0.01
    54:         self._pid_kd: float = 0.01
    55:         # PID state
    56:         self._pid_i: float = 0.0
    57:         self._delta_p: float = 0.0
    58:         self._cost_d: float = 0.0
    59:         self._cost_ds: deque = deque(maxlen=10)
    60:         self._cost_ds.append(0.0)
    61:         self._lagrangian_multiplier: float = 0.0
    62: 
    63:     def _init_log(self) -> None:
    64:         super()._init_log()
    65:         self._logger.register_key('Metrics/LagrangeMultiplier', min_and_max=True)
    66: 
    67:     def _update(self) -> None:
    68:         Jc = self._logger.get_stats('Metrics/EpCost')[0]
    69:         assert not np.isnan(Jc), 'cost is nan'
    70:         # PID update
    71:         delta = float(Jc - self._cost_limit)
    72:         self._pid_i = max(0.0, self._pid_i + delta * self._pid_ki)
    73:         self._delta_p = 0.95 * self._delta_p + 0.05 * delta
    74:         self._cost_d = 0.95 * self._cost_d + 0.05 * float(Jc)
    75:         pid_d = max(0.0, self._cost_d - self._cost_ds[0])
    76:         pid_o = self._pid_kp * self._delta_p + self._pid_i + self._pid_kd * pid_d
    77:         self._lagrangian_multiplier = max(0.0, pid_o)
    78:         self._cost_ds.append(self._cost_d)
    79:         super()._update()
    80:         self._logger.store({'Metrics/LagrangeMultiplier': self._lagrangian_multiplier})
    81: 
    82:     def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
    83:         """PID Lagrangian: combine advantages using PID-controlled multiplier."""
    84:         penalty = self._lagrangian_multiplier
    85:         return (adv_r - penalty * adv_c) / (1 + penalty)
    86: 
    87:     # ===============================================================
    88:     # FIXED: Training loop with MLS-Bench metrics reporting
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

## Time Budget

You have **5 hours** of wall-clock time before submission, covering
everything you do here: reading the code, editing it, and any trial runs
you launch.

Good luck.
