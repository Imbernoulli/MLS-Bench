# MLS-Bench: rl-value-atari

# Online RL: Value-Based Methods for Visual Control (Atari)

## Research Question
Design and implement a value-based RL algorithm for visual / Atari
environments using CNN feature extraction. Your code goes in
`custom_value_atari.py`. Several reference implementations are provided
as read-only `*.edit.py` baselines.

## Background
Atari games require learning from raw pixel observations (84x84
grayscale, 4 stacked frames). Value-based methods must learn an
effective visual representation alongside Q-value estimation, handle
high-dimensional observations, deal with sparse / delayed rewards, and
use experience replay efficiently. Different design points address
these via double targets, dueling decomposition, distributional value
functions, or quantile critics.

Reference baselines spanning the design space:
- **QR-DQN** — Dabney et al., "Distributional Reinforcement Learning
  with Quantile Regression" (arXiv:1710.10044, AAAI 2018).
  Quantile-regression distributional critic with default 200 quantiles
  trained with the Huber quantile loss.
- **C51** — Bellemare, Dabney and Munos, "A Distributional Perspective
  on Reinforcement Learning" (arXiv:1707.06887, ICML 2017). Categorical
  distributional value function with default 51 atoms over `[-10, 10]`.
- **Double DQN** — van Hasselt, Guez and Silver, "Deep Reinforcement
  Learning with Double Q-learning" (arXiv:1509.06461, AAAI 2016).
  Decouples action selection from action evaluation in the TD target.

## Constraints
- Network architecture dimensions are FIXED and cannot be modified.
- Total parameter count is enforced at runtime; the contribution must
  be algorithmic (head design, target construction, TD loss,
  exploration, replay usage) rather than capacity.
- Do **not** simply copy a reference implementation with minor changes.

## Evaluation
Trained and evaluated on multiple Atari games including Breakout, Pong
and BeamRider within a fixed interaction budget using the benchmark
Atari wrappers. Metric: mean episodic return over evaluation episodes
(higher is better). Strong methods should improve across games rather
than tuning to a single title.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/cleanrl/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to score zero.

- `cleanrl/cleanrl/custom_value_atari.py`
- editable lines **186–249**




## Readable Context


### `cleanrl/cleanrl/custom_value_atari.py`  [EDITABLE — lines 186–249 only]

```python
     1: # Custom value-based RL algorithm for Atari -- MLS-Bench
     2: #
     3: # EDITABLE section: QNetwork head and ValueAlgorithm classes.
     4: # FIXED sections: everything else (config, env, buffer, encoder, eval, training loop).
     5: import os
     6: import random
     7: import time
     8: from dataclasses import dataclass
     9: 
    10: import gymnasium as gym
    11: import numpy as np
    12: import torch
    13: import torch.nn as nn
    14: import torch.nn.functional as F
    15: import torch.optim as optim
    16: import tyro
    17: 
    18: from cleanrl_utils.atari_wrappers import (
    19:     ClipRewardEnv,
    20:     EpisodicLifeEnv,
    21:     FireResetEnv,
    22:     MaxAndSkipEnv,
    23:     NoopResetEnv,
    24: )
    25: from cleanrl_utils.buffers import ReplayBuffer
    26: 
    27: 
    28: # =====================================================================
    29: # FIXED: Configuration
    30: # =====================================================================
    31: @dataclass
    32: class Args:
    33:     exp_name: str = os.path.basename(__file__)[: -len(".py")]
    34:     """the name of this experiment"""
    35:     seed: int = 1
    36:     """seed of the experiment"""
    37:     torch_deterministic: bool = True
    38:     """if toggled, `torch.backends.cudnn.deterministic=False`"""
    39:     cuda: bool = True
    40:     """if toggled, cuda will be enabled by default"""
    41: 
    42:     # Algorithm specific arguments
    43:     env_id: str = "BreakoutNoFrameskip-v4"
    44:     """the id of the environment"""
    45:     total_timesteps: int = 5000000
    46:     """total timesteps of the experiments"""
    47:     learning_rate: float = 1e-4
    48:     """the learning rate of the optimizer"""
    49:     buffer_size: int = 1000000
    50:     """the replay memory buffer size"""
    51:     gamma: float = 0.99
    52:     """the discount factor gamma"""
    53:     tau: float = 1.0
    54:     """the target network update rate"""
    55:     target_network_frequency: int = 1000
    56:     """the timesteps it takes to update the target network"""
    57:     batch_size: int = 32
    58:     """the batch size of sample from the replay memory"""
    59:     start_e: float = 1
    60:     """the starting epsilon for exploration"""
    61:     end_e: float = 0.01
    62:     """the ending epsilon for exploration"""
    63:     exploration_fraction: float = 0.10
    64:     """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    65:     learning_starts: int = 80000
    66:     """timestep to start learning"""
    67:     train_frequency: int = 4
    68:     """the frequency of training"""
    69:     eval_freq: int = 100000
    70:     """evaluation frequency (timesteps)"""
    71:     eval_episodes: int = 10
    72:     """number of evaluation episodes"""
    73: 
    74: 
    75: # =====================================================================
    76: # FIXED: Environment setup
    77: # =====================================================================
    78: def make_env(env_id, seed):
    79:     """Create a training environment with the full Atari wrapper stack."""
    80:     def thunk():
    81:         env = gym.make(env_id)
    82:         env = gym.wrappers.RecordEpisodeStatistics(env)
    83:         env = NoopResetEnv(env, noop_max=30)
    84:         env = MaxAndSkipEnv(env, skip=4)
    85:         env = EpisodicLifeEnv(env)
    86:         if "FIRE" in env.unwrapped.get_action_meanings():
    87:             env = FireResetEnv(env)
    88:         env = ClipRewardEnv(env)
    89:         env = gym.wrappers.ResizeObservation(env, (84, 84))
    90:         env = gym.wrappers.GrayScaleObservation(env)
    91:         env = gym.wrappers.FrameStack(env, 4)
    92:         env.action_space.seed(seed)
    93:         return env
    94:     return thunk
    95: 
    96: 
    97: def make_eval_env(env_id, seed):
    98:     """Create an evaluation environment (no EpisodicLifeEnv, no ClipRewardEnv)."""
    99:     def thunk():
   100:         env = gym.make(env_id)
   101:         env = gym.wrappers.RecordEpisodeStatistics(env)
   102:         env = NoopResetEnv(env, noop_max=30)
   103:         env = MaxAndSkipEnv(env, skip=4)
   104:         if "FIRE" in env.unwrapped.get_action_meanings():
   105:             env = FireResetEnv(env)
   106:         env = gym.wrappers.ResizeObservation(env, (84, 84))
   107:         env = gym.wrappers.GrayScaleObservation(env)
   108:         env = gym.wrappers.FrameStack(env, 4)
   109:         env.action_space.seed(seed)
   110:         return env
   111:     return thunk
   112: 
   113: 
   114: # =====================================================================
   115: # FIXED: Replay Buffer (uses cleanrl_utils.buffers.ReplayBuffer)
   116: # =====================================================================
   117: # The ReplayBuffer is instantiated in the training loop below using
   118: # cleanrl_utils.buffers.ReplayBuffer with optimize_memory_usage=True.
   119: 
   120: 
   121: # =====================================================================
   122: # FIXED: Utilities
   123: # =====================================================================
   124: def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
   125:     slope = (end_e - start_e) / duration
   126:     return max(slope * t + start_e, end_e)
   127: 
   128: 
   129: @torch.no_grad()
   130: def eval_qnetwork(env_id, algorithm, device, n_episodes, seed):
   131:     """Evaluate value algorithm for n_episodes in a fresh eval env; returns array of episode rewards."""
   132:     eval_envs = gym.vector.SyncVectorEnv([make_eval_env(env_id, seed)])
   133:     episode_rewards = []
   134:     obs, _ = eval_envs.reset(seed=seed)
   135:     while len(episode_rewards) < n_episodes:
   136:         q_values = algorithm.q_network(torch.Tensor(obs).to(device))
   137:         actions = torch.argmax(q_values, dim=1).cpu().numpy()
   138:         obs, rewards, terminations, truncations, infos = eval_envs.step(actions)
   139:         if "final_info" in infos:
   140:             for info in infos["final_info"]:
   141:                 if info and "episode" in info:
   142:                     episode_rewards.append(float(info["episode"]["r"]))
   143:     eval_envs.close()
   144:     return np.asarray(episode_rewards[:n_episodes])
   145: 
   146: 
   147: # =====================================================================
   148: # FIXED: Nature DQN Encoder (network capacity is controlled here)
   149: # =====================================================================
   150: ENCODER_FEATURE_DIM = 512
   151: 
   152: 
   153: class NatureDQNEncoder(nn.Module):
   154:     """Nature DQN CNN encoder (Mnih et al. 2015).
   155: 
   156:     Input: (B, 4, 84, 84) uint8 frames
   157:     Output: (B, 512) feature vector
   158: 
   159:     All algorithms share this backbone. Only the head (defined in the
   160:     EDITABLE section) may differ.
   161:     """
   162: 
   163:     def __init__(self):
   164:         super().__init__()
   165:         self.conv = nn.Sequential(
   166:             nn.Conv2d(4, 32, 8, stride=4),
   167:             nn.ReLU(),
   168:             nn.Conv2d(32, 64, 4, stride=2),
   169:             nn.ReLU(),
   170:             nn.Conv2d(64, 64, 3, stride=1),
   171:             nn.ReLU(),
   172:             nn.Flatten(),
   173:         )
   174:         self.fc = nn.Sequential(
   175:             nn.Linear(3136, ENCODER_FEATURE_DIM),
   176:             nn.ReLU(),
   177:         )
   178: 
   179:     def forward(self, x):
   180:         return self.fc(self.conv(x / 255.0))
   181: 
   182: 
   183: # =====================================================================
   184: # EDITABLE: QNetwork head and ValueAlgorithm
   185: # =====================================================================
   186: class QNetwork(nn.Module):
   187:     """Q-network: NatureDQNEncoder (fixed) + head. Output: Q-values per action.
   188: 
   189:     The encoder is FIXED (Nature DQN CNN -> 512-dim features). Only the
   190:     head layer(s) on top of the 512-dim features may be changed.
   191:     """
   192: 
   193:     def __init__(self, envs):
   194:         super().__init__()
   195:         n_actions = envs.single_action_space.n
   196:         self.encoder = NatureDQNEncoder()
   197:         self.head = nn.Linear(ENCODER_FEATURE_DIM, n_actions)
   198: 
   199:     def forward(self, x):
   200:         features = self.encoder(x)
   201:         return self.head(features)
   202: 
   203: 
   204: class ValueAlgorithm:
   205:     """Value-based algorithm for Atari -- implement your approach here.
   206: 
   207:     The training loop calls:
   208:         algorithm = ValueAlgorithm(envs, device, args)
   209:         action = algorithm.select_action(obs, epsilon)
   210:         metrics = algorithm.update(batch, global_step)
   211:         eval_qnetwork(env_id, algorithm, device, ...)
   212: 
   213:     You MUST set self.q_network and self.target_network to nn.Module instances.
   214: 
   215:     Available classes:
   216:         NatureDQNEncoder (fixed) -- Nature DQN CNN encoder, -> 512-dim features
   217:         QNetwork         (editable) -- NatureDQNEncoder + head
   218:     ENCODER_FEATURE_DIM = 512 (feature dimension from NatureDQNEncoder)
   219: 
   220:     Available utilities (fixed): linear_schedule
   221:     """
   222: 
   223:     def __init__(self, envs, device, args):
   224:         self.device = device
   225:         self.gamma = args.gamma
   226:         self.tau = args.tau
   227:         self.target_network_frequency = args.target_network_frequency
   228: 
   229:         self.q_network = QNetwork(envs).to(device)
   230:         self.target_network = QNetwork(envs).to(device)
   231:         self.target_network.load_state_dict(self.q_network.state_dict())
   232:         self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate)
   233: 
   234:     def select_action(self, obs, epsilon):
   235:         """Epsilon-greedy action selection."""
   236:         if random.random() < epsilon:
   237:             return np.array([self.q_network.head.out_features])  # placeholder
   238:         q_values = self.q_network(torch.Tensor(obs).to(self.device))
   239:         return torch.argmax(q_values, dim=1).cpu().numpy()
   240: 
   241:     def update(self, batch, global_step):
   242:         """Single gradient update. Returns a dict of scalar metrics.
   243: 
   244:         batch: cleanrl ReplayBuffer sample with .observations, .next_observations,
   245:                .actions, .rewards, .dones
   246: 
   247:         TODO: implement your value-based RL algorithm here.
   248:         """
   249:         return {"td_loss": 0.0, "q_values": 0.0}
   250: 
   251: 
   252: # =====================================================================
   253: # FIXED: Parameter count assertion
   254: # =====================================================================
   255: def _check_param_budget(q_network, n_actions):
   256:     """Ensure the Q-network does not exceed the parameter budget.
   257: 
   258:     The budget is the NatureDQNEncoder params + a generous head allowance.
   259:     This prevents capacity hacking by adding hidden layers.
   260:     """
   261:     encoder_params = sum(p.numel() for p in NatureDQNEncoder().parameters())
   262:     # Largest head: QR-DQN with 200 quantiles: n_actions * 200 outputs
   263:     max_head_output = n_actions * 200
   264:     max_head_params = ENCODER_FEATURE_DIM * max_head_output + max_head_output
   265:     max_total = int((encoder_params + max_head_params) * 1.05)
   266:     actual = sum(p.numel() for p in q_network.parameters())
   267:     print(
   268:         f"QNetwork parameters: {actual:,} / {max_total:,} "
   269:         f"(1.05x largest baseline, informational only)",
   270:         flush=True,
   271:     )
   272: 
   273: 
   274: # =====================================================================
   275: # FIXED: Training loop
   276: # =====================================================================
   277: if __name__ == "__main__":
   278:     args = tyro.cli(Args)
   279:     run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
   280: 
   281:     # Seeding
   282:     random.seed(args.seed)
   283:     np.random.seed(args.seed)
   284:     torch.manual_seed(args.seed)
   285:     torch.backends.cudnn.deterministic = args.torch_deterministic
   286: 
   287:     device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
   288: 
   289:     # Environment setup
   290:     envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed)])
   291:     assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
   292: 
   293:     # Algorithm
   294:     algorithm = ValueAlgorithm(envs, device, args)
   295: 
   296:     # Parameter budget check
   297:     _check_param_budget(algorithm.q_network, envs.single_action_space.n)
   298: 
   299:     # Replay buffer
   300:     rb = ReplayBuffer(
   301:         args.buffer_size,
   302:         envs.single_observation_space,
   303:         envs.single_action_space,
   304:         device,
   305:         optimize_memory_usage=True,
   306:         handle_timeout_termination=False,
   307:     )
   308: 
   309:     start_time = time.time()
   310:     obs, _ = envs.reset(seed=args.seed)
   311: 
   312:     for global_step in range(args.total_timesteps):
   313:         # Epsilon-greedy action selection
   314:         epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
   315:         if random.random() < epsilon:
   316:             actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
   317:         else:
   318:             actions = algorithm.select_action(obs, epsilon=0.0)
   319: 
   320:         # Environment step
   321:         next_obs, rewards, terminations, truncations, infos = envs.step(actions)
   322: 
   323:         if "final_info" in infos:
   324:             for info in infos["final_info"]:
   325:                 if info and "episode" in info:
   326:                     print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
   327:                     break
   328: 
   329:         # Handle truncation
   330:         real_next_obs = next_obs.copy()
   331:         for idx, trunc in enumerate(truncations):
   332:             if trunc:
   333:                 real_next_obs[idx] = infos["final_observation"][idx]
   334:         rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
   335:         obs = next_obs
   336: 
   337:         # Training
   338:         if global_step > args.learning_starts:
   339:             if global_step % args.train_frequency == 0:
   340:                 batch = rb.sample(args.batch_size)
   341:                 log_dict = algorithm.update(batch, global_step)
   342: 
   343:                 if global_step % 1000 == 0:
   344:                     metrics_str = " ".join(
   345:                         f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
   346:                         for k, v in log_dict.items()
   347:                     )
   348:                     print(f"TRAIN_METRICS step={global_step} {metrics_str}", flush=True)
   349: 
   350:         # Evaluation
   351:         if (global_step + 1) % args.eval_freq == 0:
   352:             eval_returns = eval_qnetwork(
   353:                 args.env_id, algorithm, device,
   354:                 n_episodes=args.eval_episodes, seed=args.seed + 1000,
   355:             )
   356:             mean_return = eval_returns.mean()
   357:             print(f"Eval episodic_return: {mean_return:.2f}", flush=True)
   358: 
   359:     envs.close()
```




## How You Will Be Evaluated

After you finish, evaluation runs a fixed set of scripts and aggregates the
metrics they emit. These scripts are **not** in your workspace — you cannot
read or modify them. The labels below indicate what each evaluation tests:

- **breakout-v4** — wall-clock budget `24:00:00`, compute share `0.4`
- **seaquest-v4** — wall-clock budget `24:00:00`, compute share `0.4`
- **pong-v4** — wall-clock budget `24:00:00`, compute share `0.4`


Scoring uses the same `combined_score` aggregation as the MLS-Bench
leaderboard. Multiple seeds are averaged.

## Parameter Budget

This task enforces a parameter-count cap. Your edits will be rejected if
the resulting model exceeds **1.05×** the strongest
baseline's parameter count. The check runs automatically inside the eval
scripts — you don't need to invoke it.

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `qr_dqn` baseline — editable region  [READ-ONLY — reference implementation]

In `cleanrl/cleanrl/custom_value_atari.py`:

```python
Lines 186–273:
   183: # =====================================================================
   184: # EDITABLE: QNetwork head and ValueAlgorithm
   185: # =====================================================================
   186: class QNetwork(nn.Module):
   187:     """QR-DQN quantile Q-network: NatureDQNEncoder (fixed) + quantile head."""
   188: 
   189:     def __init__(self, envs, n_quantiles=200):
   190:         super().__init__()
   191:         self.n_quantiles = n_quantiles
   192:         self.n = envs.single_action_space.n
   193:         self.encoder = NatureDQNEncoder()
   194:         self.head = nn.Linear(ENCODER_FEATURE_DIM, self.n * n_quantiles)
   195: 
   196:     def forward(self, x):
   197:         """Return Q-values as mean of quantile values per action."""
   198:         features = self.encoder(x)
   199:         quantiles = self.head(features).view(len(x), self.n, self.n_quantiles)
   200:         q_values = quantiles.mean(dim=2)
   201:         return q_values
   202: 
   203:     def get_quantiles(self, x):
   204:         """Return raw quantile values: [batch, n_actions, n_quantiles]."""
   205:         features = self.encoder(x)
   206:         return self.head(features).view(len(x), self.n, self.n_quantiles)
   207: 
   208: 
   209: class ValueAlgorithm:
   210:     """QR-DQN -- Quantile Regression DQN with distributional value learning."""
   211: 
   212:     def __init__(self, envs, device, args):
   213:         self.device = device
   214:         self.gamma = args.gamma
   215:         self.target_network_frequency = args.target_network_frequency
   216:         self.n_quantiles = 200
   217:         self.kappa = 1.0  # Huber loss threshold
   218: 
   219:         self.q_network = QNetwork(envs, n_quantiles=self.n_quantiles).to(device)
   220:         self.target_network = QNetwork(envs, n_quantiles=self.n_quantiles).to(device)
   221:         self.target_network.load_state_dict(self.q_network.state_dict())
   222:         self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate, eps=0.01 / args.batch_size)
   223: 
   224:         # Fixed quantile midpoints: tau_i = (2i - 1) / (2N) for i = 1, ..., N
   225:         self.tau = torch.arange(1, self.n_quantiles + 1, dtype=torch.float32, device=device)
   226:         self.tau = (2 * self.tau - 1) / (2 * self.n_quantiles)
   227: 
   228:     def select_action(self, obs, epsilon):
   229:         """Greedy action selection using mean of quantile values."""
   230:         q_values = self.q_network(torch.Tensor(obs).to(self.device))
   231:         return torch.argmax(q_values, dim=1).cpu().numpy()
   232: 
   233:     def update(self, batch, global_step):
   234:         """QR-DQN update: quantile Huber loss."""
   235:         with torch.no_grad():
   236:             # Get quantile values for next state from target network
   237:             next_quantiles = self.target_network.get_quantiles(batch.next_observations)  # [batch, n_actions, N]
   238:             next_q = next_quantiles.mean(dim=2)  # [batch, n_actions]
   239:             next_actions = next_q.argmax(dim=1)  # [batch]
   240:             # Select quantiles for best actions
   241:             next_quantiles_best = next_quantiles[torch.arange(len(batch.next_observations)), next_actions]  # [batch, N]
   242:             # Compute target quantile values
   243:             target_quantiles = batch.rewards + self.gamma * next_quantiles_best * (1 - batch.dones)
   244: 
   245:         # Get current quantile values for taken actions
   246:         current_quantiles_all = self.q_network.get_quantiles(batch.observations)  # [batch, n_actions, N]
   247:         current_quantiles = current_quantiles_all[torch.arange(len(batch.observations)), batch.actions.flatten()]  # [batch, N]
   248: 
   249:         # Quantile Huber loss
   250:         # Pairwise TD errors: [batch, N (pred), N (target)]
   251:         td_errors = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)
   252: 
   253:         # Huber loss element-wise
   254:         abs_td = td_errors.abs()
   255:         huber = torch.where(abs_td <= self.kappa,
   256:                             0.5 * td_errors ** 2,
   257:                             self.kappa * (abs_td - 0.5 * self.kappa))
   258: 
   259:         # Asymmetric weighting by quantile level
   260:         tau = self.tau.view(1, -1, 1)
   261:         quantile_weights = torch.abs(tau - (td_errors < 0).float())
   262:         loss = (quantile_weights * huber / self.kappa).sum(dim=2).mean(dim=1).mean()
   263: 
   264:         self.optimizer.zero_grad()
   265:         loss.backward()
   266:         self.optimizer.step()
   267: 
   268:         # Hard target update
   269:         if global_step % self.target_network_frequency == 0:
   270:             self.target_network.load_state_dict(self.q_network.state_dict())
   271: 
   272:         q_values = current_quantiles.mean(dim=1)
   273:         return {"td_loss": loss.item(), "q_values": q_values.mean().item()}
   274: 
   275: 
   276: # =====================================================================
```

### `c51` baseline — editable region  [READ-ONLY — reference implementation]

In `cleanrl/cleanrl/custom_value_atari.py`:

```python
Lines 186–271:
   183: # =====================================================================
   184: # EDITABLE: QNetwork head and ValueAlgorithm
   185: # =====================================================================
   186: class QNetwork(nn.Module):
   187:     """C51 distributional Q-network: NatureDQNEncoder (fixed) + distributional head."""
   188: 
   189:     def __init__(self, envs, n_atoms=51, v_min=-10, v_max=10):
   190:         super().__init__()
   191:         self.n_atoms = n_atoms
   192:         self.n = envs.single_action_space.n
   193:         self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=n_atoms))
   194:         self.encoder = NatureDQNEncoder()
   195:         self.head = nn.Linear(ENCODER_FEATURE_DIM, self.n * n_atoms)
   196: 
   197:     def forward(self, x):
   198:         """Return Q-values (expected values under the learned distribution)."""
   199:         features = self.encoder(x)
   200:         logits = self.head(features)
   201:         pmfs = torch.softmax(logits.view(len(x), self.n, self.n_atoms), dim=2)
   202:         q_values = (pmfs * self.atoms).sum(2)
   203:         return q_values
   204: 
   205:     def get_action(self, x, action=None):
   206:         """Return (action, pmf_for_action). If action is None, use greedy."""
   207:         features = self.encoder(x)
   208:         logits = self.head(features)
   209:         pmfs = torch.softmax(logits.view(len(x), self.n, self.n_atoms), dim=2)
   210:         q_values = (pmfs * self.atoms).sum(2)
   211:         if action is None:
   212:             action = torch.argmax(q_values, 1)
   213:         return action, pmfs[torch.arange(len(x)), action]
   214: 
   215: 
   216: class ValueAlgorithm:
   217:     """C51 -- Categorical Distributional DQN."""
   218: 
   219:     def __init__(self, envs, device, args):
   220:         self.device = device
   221:         self.gamma = args.gamma
   222:         # CleanRL's c51_atari.py uses a slower target refresh and larger
   223:         # learning rate than DQN. Keeping the template/DQN values makes C51
   224:         # systematically underperform on harder Atari games such as Seaquest.
   225:         self.target_network_frequency = 10000
   226:         self.n_atoms = 51
   227:         self.v_min = -10.0
   228:         self.v_max = 10.0
   229: 
   230:         self.q_network = QNetwork(envs, n_atoms=self.n_atoms, v_min=self.v_min, v_max=self.v_max).to(device)
   231:         self.target_network = QNetwork(envs, n_atoms=self.n_atoms, v_min=self.v_min, v_max=self.v_max).to(device)
   232:         self.target_network.load_state_dict(self.q_network.state_dict())
   233:         self.optimizer = optim.Adam(self.q_network.parameters(), lr=2.5e-4, eps=0.01 / args.batch_size)
   234: 
   235:     def select_action(self, obs, epsilon):
   236:         """Greedy action selection using distributional Q-values."""
   237:         action, _ = self.q_network.get_action(torch.Tensor(obs).to(self.device))
   238:         return action.cpu().numpy()
   239: 
   240:     def update(self, batch, global_step):
   241:         """C51 distributional update: categorical projection + cross-entropy loss."""
   242:         with torch.no_grad():
   243:             _, next_pmfs = self.target_network.get_action(batch.next_observations)
   244:             next_atoms = batch.rewards + self.gamma * self.target_network.atoms * (1 - batch.dones)
   245:             # Projection
   246:             delta_z = self.target_network.atoms[1] - self.target_network.atoms[0]
   247:             tz = next_atoms.clamp(self.v_min, self.v_max)
   248:             b = (tz - self.v_min) / delta_z
   249:             l = b.floor().clamp(0, self.n_atoms - 1)
   250:             u = b.ceil().clamp(0, self.n_atoms - 1)
   251:             # Handle case where b is exactly an integer
   252:             d_m_l = (u + (l == u).float() - b) * next_pmfs
   253:             d_m_u = (b - l) * next_pmfs
   254:             target_pmfs = torch.zeros_like(next_pmfs)
   255:             for i in range(target_pmfs.size(0)):
   256:                 target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
   257:                 target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])
   258: 
   259:         _, old_pmfs = self.q_network.get_action(batch.observations, batch.actions.flatten())
   260:         loss = (-(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean()
   261: 
   262:         self.optimizer.zero_grad()
   263:         loss.backward()
   264:         self.optimizer.step()
   265: 
   266:         # Hard target update
   267:         if global_step % self.target_network_frequency == 0:
   268:             self.target_network.load_state_dict(self.q_network.state_dict())
   269: 
   270:         old_val = (old_pmfs * self.q_network.atoms).sum(1)
   271:         return {"td_loss": loss.item(), "q_values": old_val.mean().item()}
   272: 
   273: 
   274: # =====================================================================
```

### `double_dqn` baseline — editable region  [READ-ONLY — reference implementation]

In `cleanrl/cleanrl/custom_value_atari.py`:

```python
Lines 186–245:
   183: # =====================================================================
   184: # EDITABLE: QNetwork head and ValueAlgorithm
   185: # =====================================================================
   186: class QNetwork(nn.Module):
   187:     """Q-network: NatureDQNEncoder (fixed) + head. Output: Q-values per action.
   188: 
   189:     The encoder is FIXED (Nature DQN CNN -> 512-dim features). Only the
   190:     head layer(s) on top of the 512-dim features may be changed.
   191:     """
   192: 
   193:     def __init__(self, envs):
   194:         super().__init__()
   195:         n_actions = envs.single_action_space.n
   196:         self.encoder = NatureDQNEncoder()
   197:         self.head = nn.Linear(ENCODER_FEATURE_DIM, n_actions)
   198: 
   199:     def forward(self, x):
   200:         features = self.encoder(x)
   201:         return self.head(features)
   202: 
   203: 
   204: class ValueAlgorithm:
   205:     """DoubleDQN -- Double Deep Q-Network with hard target updates."""
   206: 
   207:     def __init__(self, envs, device, args):
   208:         self.device = device
   209:         self.gamma = args.gamma
   210:         self.tau = args.tau
   211:         self.target_network_frequency = args.target_network_frequency
   212: 
   213:         self.q_network = QNetwork(envs).to(device)
   214:         self.target_network = QNetwork(envs).to(device)
   215:         self.target_network.load_state_dict(self.q_network.state_dict())
   216:         self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate)
   217: 
   218:     def select_action(self, obs, epsilon):
   219:         """Epsilon-greedy action selection."""
   220:         q_values = self.q_network(torch.Tensor(obs).to(self.device))
   221:         return torch.argmax(q_values, dim=1).cpu().numpy()
   222: 
   223:     def update(self, batch, global_step):
   224:         """DoubleDQN update: online net selects action, target net evaluates."""
   225:         with torch.no_grad():
   226:             # Double Q-learning: online net selects best action, target net evaluates it
   227:             best_actions = self.q_network(batch.next_observations).argmax(dim=1, keepdim=True)
   228:             target_q = self.target_network(batch.next_observations).gather(1, best_actions).squeeze()
   229:             td_target = batch.rewards.flatten() + self.gamma * target_q * (1 - batch.dones.flatten())
   230: 
   231:         old_val = self.q_network(batch.observations).gather(1, batch.actions).squeeze()
   232:         loss = F.mse_loss(td_target, old_val)
   233: 
   234:         self.optimizer.zero_grad()
   235:         loss.backward()
   236:         self.optimizer.step()
   237: 
   238:         # Hard target update
   239:         if global_step % self.target_network_frequency == 0:
   240:             for target_param, q_param in zip(self.target_network.parameters(), self.q_network.parameters()):
   241:                 target_param.data.copy_(
   242:                     self.tau * q_param.data + (1.0 - self.tau) * target_param.data
   243:                 )
   244: 
   245:         return {"td_loss": loss.item(), "q_values": old_val.mean().item()}
   246: 
   247: 
   248: # =====================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
