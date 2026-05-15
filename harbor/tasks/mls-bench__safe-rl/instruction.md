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


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/omnisafe/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to score zero.

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
    76:         """Training loop with TRAIN_METRICS and TEST_METRICS output."""
    77:         start_time = time.time()
    78:         self._logger.log('INFO: Start training')
    79: 
    80:         for epoch in range(self._cfgs.train_cfgs.epochs):
    81:             epoch_time = time.time()
    82: 
    83:             rollout_time = time.time()
    84:             self._env.rollout(
    85:                 steps_per_epoch=self._steps_per_epoch,
    86:                 agent=self._actor_critic,
    87:                 buffer=self._buf,
    88:                 logger=self._logger,
    89:             )
    90:             self._logger.store({'Time/Rollout': time.time() - rollout_time})
    91: 
    92:             update_time = time.time()
    93:             self._update()
    94:             self._logger.store({'Time/Update': time.time() - update_time})
    95: 
    96:             if self._cfgs.model_cfgs.exploration_noise_anneal:
    97:                 self._actor_critic.annealing(epoch)
    98: 
    99:             if self._cfgs.model_cfgs.actor.lr is not None:
   100:                 self._actor_critic.actor_scheduler.step()
   101: 
   102:             self._logger.store(
   103:                 {
   104:                     'TotalEnvSteps': (epoch + 1) * self._cfgs.algo_cfgs.steps_per_epoch,
   105:                     'Time/FPS': self._cfgs.algo_cfgs.steps_per_epoch / (time.time() - epoch_time),
   106:                     'Time/Total': (time.time() - start_time),
   107:                     'Time/Epoch': (time.time() - epoch_time),
   108:                     'Train/Epoch': epoch,
   109:                     'Train/LR': (
   110:                         0.0
   111:                         if self._cfgs.model_cfgs.actor.lr is None
   112:                         else self._actor_critic.actor_scheduler.get_last_lr()[0]
   113:                     ),
   114:                 },
   115:             )
   116: 
   117:             self._logger.dump_tabular()
   118: 
   119:             # -- MLS-Bench: TRAIN_METRICS --
   120:             _ep_ret = self._logger.get_stats('Metrics/EpRet')[0]
   121:             _ep_cost = self._logger.get_stats('Metrics/EpCost')[0]
   122:             _ep_len = self._logger.get_stats('Metrics/EpLen')[0]
   123:             print(
   124:                 f'TRAIN_METRICS epoch={epoch} '
   125:                 f'ep_ret={_ep_ret:.4f} ep_cost={_ep_cost:.4f} '
   126:                 f'ep_len={_ep_len:.1f}',
   127:                 flush=True,
   128:             )
   129: 
   130:             if (epoch + 1) % self._cfgs.logger_cfgs.save_model_freq == 0 or (
   131:                 epoch + 1
   132:             ) == self._cfgs.train_cfgs.epochs:
   133:                 self._logger.torch_save()
   134: 
   135:         ep_ret = self._logger.get_stats('Metrics/EpRet')[0]
   136:         ep_cost = self._logger.get_stats('Metrics/EpCost')[0]
   137:         ep_len = self._logger.get_stats('Metrics/EpLen')[0]
   138: 
   139:         # -- MLS-Bench: TEST_METRICS --
   140:         print(
   141:             f'TEST_METRICS ep_ret={ep_ret:.4f} ep_cost={ep_cost:.4f} '
   142:             f'ep_len={ep_len:.1f}',
   143:             flush=True,
   144:         )
   145: 
   146:         self._logger.close()
   147:         self._env.close()
   148: 
   149:         return ep_ret, ep_cost, ep_len
```




## How You Will Be Evaluated

After you finish, evaluation runs a fixed set of scripts and aggregates the
metrics they emit. These scripts are **not** in your workspace — you cannot
read or modify them. The labels below indicate what each evaluation tests:

- **SafetyPointGoal1-v0** — wall-clock budget `6:00:00`, compute share `0.33`
- **SafetyCarGoal1-v0** — wall-clock budget `6:00:00`, compute share `0.33`
- **SafetyPointButton1-v0** — wall-clock budget `6:00:00`, compute share `0.33`


Scoring uses the same `combined_score` aggregation as the MLS-Bench
leaderboard. Multiple seeds are averaged.



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

Good luck.
