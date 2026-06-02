# MLS-Bench: rl-value-discrete

# Online RL: Value-Based Methods for Discrete Control

## Research Question
Design and implement a value-based RL algorithm for discrete action
spaces. Your code goes in `custom_value_discrete.py`. Several reference
implementations are provided as read-only `*.edit.py` baselines.

## Background
Value-based methods estimate Q-values `Q(s, a)` for each state-action
pair and derive a policy by acting greedily (or epsilon-greedily) with
respect to those estimates. In small to medium discrete-control tasks,
the key algorithmic challenges are overestimation bias under
bootstrapped targets, unstable value learning, exploration scheduling,
and representing uncertainty or full return distributions.

Reference baselines spanning the design space:
- **QR-DQN** — Dabney et al., "Distributional Reinforcement Learning
  with Quantile Regression" (arXiv:1710.10044, AAAI 2018).
  Quantile-regression distributional critic with default 200 quantiles
  trained with the Huber quantile loss.
- **Dueling DQN** — Wang et al., "Dueling Network Architectures for
  Deep Reinforcement Learning" (arXiv:1511.06581, ICML 2016). Splits
  the head into state value and action advantage streams.
- **C51** — Bellemare, Dabney and Munos, "A Distributional Perspective
  on Reinforcement Learning" (arXiv:1707.06887, ICML 2017). Categorical
  distributional critic with default 51 atoms over `[-10, 10]`.

## Constraints
- Network architecture dimensions are FIXED and cannot be modified.
- Total parameter count is enforced at runtime; the contribution must
  be algorithmic (head design, target construction, TD loss,
  exploration, replay usage) rather than encoder capacity.
- Do **not** simply copy a reference implementation with minor changes.

## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/cleanrl/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `cleanrl/cleanrl/custom_value_discrete.py`
- editable lines **174–242**




## Readable Context


### `cleanrl/cleanrl/custom_value_discrete.py`  [EDITABLE — lines 174–242 only]

```python
     1: # Custom value-based discrete RL algorithm for MLS-Bench
     2: #
     3: # EDITABLE section: QNetwork head and ValueAlgorithm classes.
     4: # FIXED sections: everything else (config, env, buffer, encoder, utility, training loop).
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
    18: 
    19: # =====================================================================
    20: # FIXED: Configuration
    21: # =====================================================================
    22: @dataclass
    23: class Args:
    24:     exp_name: str = os.path.basename(__file__)[: -len(".py")]
    25:     """the name of this experiment"""
    26:     seed: int = 1
    27:     """seed of the experiment"""
    28:     torch_deterministic: bool = True
    29:     """if toggled, `torch.backends.cudnn.deterministic=False`"""
    30:     cuda: bool = True
    31:     """if toggled, cuda will be enabled by default"""
    32: 
    33:     # Algorithm specific arguments
    34:     env_id: str = "CartPole-v1"
    35:     """the id of the environment"""
    36:     total_timesteps: int = 500000
    37:     """total timesteps of the experiments"""
    38:     learning_rate: float = 2.5e-4
    39:     """the learning rate of the optimizer"""
    40:     buffer_size: int = 10000
    41:     """the replay memory buffer size"""
    42:     gamma: float = 0.99
    43:     """the discount factor gamma"""
    44:     tau: float = 1.0
    45:     """the target network update rate"""
    46:     target_network_frequency: int = 500
    47:     """the timesteps it takes to update the target network"""
    48:     batch_size: int = 128
    49:     """the batch size of sample from the replay memory"""
    50:     start_e: float = 1
    51:     """the starting epsilon for exploration"""
    52:     end_e: float = 0.05
    53:     """the ending epsilon for exploration"""
    54:     exploration_fraction: float = 0.5
    55:     """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    56:     learning_starts: int = 10000
    57:     """timestep to start learning"""
    58:     train_frequency: int = 10
    59:     """the frequency of training"""
    60:     eval_freq: int = 10000
    61:     """evaluation frequency (timesteps)"""
    62:     eval_episodes: int = 10
    63:     """number of evaluation episodes"""
    64: 
    65: 
    66: # =====================================================================
    67: # FIXED: Environment setup
    68: # =====================================================================
    69: def make_env(env_id, seed):
    70:     def thunk():
    71:         env = gym.make(env_id)
    72:         env = gym.wrappers.RecordEpisodeStatistics(env)
    73:         env.action_space.seed(seed)
    74:         return env
    75:     return thunk
    76: 
    77: 
    78: # =====================================================================
    79: # FIXED: Replay Buffer
    80: # =====================================================================
    81: class SimpleReplayBuffer:
    82:     """Numpy-based replay buffer for discrete actions."""
    83: 
    84:     def __init__(self, obs_dim, max_size=10000):
    85:         self.max_size = max_size
    86:         self.ptr = 0
    87:         self.size = 0
    88:         self.obs = np.zeros((max_size, obs_dim), dtype=np.float32)
    89:         self.next_obs = np.zeros((max_size, obs_dim), dtype=np.float32)
    90:         self.actions = np.zeros((max_size,), dtype=np.int64)
    91:         self.rewards = np.zeros((max_size,), dtype=np.float32)
    92:         self.dones = np.zeros((max_size,), dtype=np.float32)
    93: 
    94:     def add(self, obs, next_obs, action, reward, done):
    95:         self.obs[self.ptr] = obs
    96:         self.next_obs[self.ptr] = next_obs
    97:         self.actions[self.ptr] = action
    98:         self.rewards[self.ptr] = reward
    99:         self.dones[self.ptr] = done
   100:         self.ptr = (self.ptr + 1) % self.max_size
   101:         self.size = min(self.size + 1, self.max_size)
   102: 
   103:     def sample(self, batch_size, device):
   104:         idx = np.random.randint(0, self.size, size=batch_size)
   105:         return (
   106:             torch.tensor(self.obs[idx], device=device),
   107:             torch.tensor(self.next_obs[idx], device=device),
   108:             torch.tensor(self.actions[idx], dtype=torch.long, device=device),
   109:             torch.tensor(self.rewards[idx], device=device),
   110:             torch.tensor(self.dones[idx], device=device),
   111:         )
   112: 
   113: 
   114: # =====================================================================
   115: # FIXED: Utilities
   116: # =====================================================================
   117: def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
   118:     """Linear epsilon schedule from start_e to end_e over duration steps."""
   119:     slope = (end_e - start_e) / duration
   120:     return max(slope * t + start_e, end_e)
   121: 
   122: 
   123: @torch.no_grad()
   124: def eval_qnetwork(env_id, q_network, device, n_episodes, seed):
   125:     """Evaluate Q-network greedily for n_episodes in a fresh env; returns array of episode rewards."""
   126:     eval_env = gym.make(env_id)
   127:     episode_rewards = []
   128:     for ep in range(n_episodes):
   129:         obs, _ = eval_env.reset(seed=seed + ep)
   130:         done = False
   131:         episode_reward = 0.0
   132:         while not done:
   133:             obs_t = torch.tensor(obs.reshape(1, -1), device=device, dtype=torch.float32)
   134:             q_values = q_network(obs_t)
   135:             action = torch.argmax(q_values, dim=1).item()
   136:             obs, reward, terminated, truncated, _ = eval_env.step(action)
   137:             done = terminated or truncated
   138:             episode_reward += reward
   139:         episode_rewards.append(episode_reward)
   140:     eval_env.close()
   141:     return np.asarray(episode_rewards)
   142: 
   143: 
   144: # =====================================================================
   145: # FIXED: MLP Encoder (network capacity is controlled here)
   146: # =====================================================================
   147: ENCODER_HIDDEN_DIMS = [120, 84]
   148: ENCODER_FEATURE_DIM = ENCODER_HIDDEN_DIMS[-1]  # 84
   149: 
   150: 
   151: class MLPEncoder(nn.Module):
   152:     """Fixed 2-layer MLP encoder: obs_dim -> 120 -> 84.
   153: 
   154:     All algorithms share this backbone. Only the head (defined in the
   155:     EDITABLE section) may differ.
   156:     """
   157: 
   158:     def __init__(self, obs_dim):
   159:         super().__init__()
   160:         self.net = nn.Sequential(
   161:             nn.Linear(obs_dim, 120),
   162:             nn.ReLU(),
   163:             nn.Linear(120, 84),
   164:             nn.ReLU(),
   165:         )
   166: 
   167:     def forward(self, obs):
   168:         return self.net(obs)
   169: 
   170: 
   171: # =====================================================================
   172: # EDITABLE: QNetwork head and ValueAlgorithm
   173: # =====================================================================
   174: class QNetwork(nn.Module):
   175:     """Q-network: MLPEncoder (fixed) + head. Output: Q-values per action (batch x n_actions).
   176: 
   177:     The encoder is FIXED (120->84 MLP). Only the head layer(s) on top of
   178:     the 84-dim features may be changed.
   179:     """
   180: 
   181:     def __init__(self, obs_dim, n_actions):
   182:         super().__init__()
   183:         self.encoder = MLPEncoder(obs_dim)
   184:         self.head = nn.Linear(ENCODER_FEATURE_DIM, n_actions)
   185: 
   186:     def forward(self, obs):
   187:         features = self.encoder(obs)
   188:         return self.head(features)
   189: 
   190: 
   191: class ValueAlgorithm:
   192:     """Value-based RL algorithm -- implement your approach here.
   193: 
   194:     The training loop calls:
   195:         algorithm = ValueAlgorithm(obs_dim, n_actions, device, args)
   196:         action = algorithm.select_action(obs, epsilon)   # during data collection
   197:         metrics = algorithm.update(batch, global_step)    # after each training step
   198:         eval_qnetwork(env_id, algorithm.q_network, ...)   # periodic evaluation
   199: 
   200:     You MUST set self.q_network to an nn.Module with forward(obs) -> Q-values.
   201: 
   202:     Available classes:
   203:         MLPEncoder (fixed) -- 2-layer MLP encoder, obs_dim -> 84-dim features
   204:         QNetwork   (editable) -- MLPEncoder + head
   205:     ENCODER_FEATURE_DIM = 84 (feature dimension from MLPEncoder)
   206: 
   207:     Available utilities (fixed): linear_schedule, eval_qnetwork
   208:     """
   209: 
   210:     def __init__(self, obs_dim, n_actions, device, args):
   211:         self.device = device
   212:         self.n_actions = n_actions
   213:         self.gamma = args.gamma
   214:         self.total_it = 0
   215: 
   216:         # Build networks -- modify or replace as needed
   217:         self.q_network = QNetwork(obs_dim, n_actions).to(device)
   218:         self.target_network = QNetwork(obs_dim, n_actions).to(device)
   219:         self.target_network.load_state_dict(self.q_network.state_dict())
   220: 
   221:         self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate)
   222: 
   223:     def select_action(self, obs, epsilon):
   224:         """Select action with epsilon-greedy exploration."""
   225:         if random.random() < epsilon:
   226:             return random.randint(0, self.n_actions - 1)
   227:         obs_t = torch.tensor(obs.reshape(1, -1), device=self.device, dtype=torch.float32)
   228:         q_values = self.q_network(obs_t)
   229:         return torch.argmax(q_values, dim=1).item()
   230: 
   231:     def update(self, batch, global_step):
   232:         """Single gradient update. Returns a dict of scalar metrics.
   233: 
   234:         batch = (obs, next_obs, actions, rewards, dones) -- torch.Tensor on device
   235: 
   236:         TODO: implement your value-based RL algorithm here.
   237:         """
   238:         self.total_it += 1
   239:         obs, next_obs, actions, rewards, dones = batch
   240: 
   241:         # Placeholder -- replace with your algorithm
   242:         return {"td_loss": 0.0, "q_values": 0.0}
   243: 
   244: 
   245: # =====================================================================
   246: # FIXED: Training loop
   247: # =====================================================================
   248: if __name__ == "__main__":
   249:     args = tyro.cli(Args)
   250:     run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
   251: 
   252:     # Seeding
   253:     random.seed(args.seed)
   254:     np.random.seed(args.seed)
   255:     torch.manual_seed(args.seed)
   256:     torch.backends.cudnn.deterministic = args.torch_deterministic
   257: 
   258:     device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
   259: 
   260:     # Environment setup
   261:     envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed)])
   262:     assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
   263: 
   264:     obs_dim = np.array(envs.single_observation_space.shape).prod()
   265:     n_actions = envs.single_action_space.n
   266: 
   267:     # Algorithm
   268:     algorithm = ValueAlgorithm(obs_dim, n_actions, device, args)
   269: 
   270:     # Replay buffer
   271:     rb = SimpleReplayBuffer(obs_dim, args.buffer_size)
   272: 
   273:     start_time = time.time()
   274:     obs, _ = envs.reset(seed=args.seed)
   275: 
   276:     for global_step in range(args.total_timesteps):
   277:         # Epsilon-greedy action selection
   278:         epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
   279:         action = algorithm.select_action(obs[0], epsilon)
   280:         actions = np.array([action])
   281: 
   282:         # Environment step
   283:         next_obs, rewards, terminations, truncations, infos = envs.step(actions)
   284: 
   285:         if "final_info" in infos:
   286:             for info in infos["final_info"]:
   287:                 if info is not None and "episode" in info:
   288:                     print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
   289:                     break
   290: 
   291:         # Handle truncation
   292:         real_next_obs = next_obs.copy()
   293:         for idx, trunc in enumerate(truncations):
   294:             if trunc:
   295:                 real_next_obs[idx] = infos["final_observation"][idx]
   296: 
   297:         rb.add(obs[0], real_next_obs[0], actions[0], rewards[0], terminations[0])
   298:         obs = next_obs
   299: 
   300:         # Training
   301:         if global_step > args.learning_starts:
   302:             if global_step % args.train_frequency == 0:
   303:                 batch = rb.sample(args.batch_size, device)
   304:                 log_dict = algorithm.update(batch, global_step)
   305: 
   306:                 if global_step % 1000 == 0:
   307:                     metrics_str = " ".join(
   308:                         f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
   309:                         for k, v in log_dict.items()
   310:                     )
   311:                     print(f"TRAIN_METRICS step={global_step} {metrics_str}", flush=True)
   312: 
   313:             # Update target network
   314:             if global_step % args.target_network_frequency == 0:
   315:                 for target_param, q_param in zip(algorithm.target_network.parameters(), algorithm.q_network.parameters()):
   316:                     target_param.data.copy_(
   317:                         args.tau * q_param.data + (1.0 - args.tau) * target_param.data
   318:                     )
   319: 
   320:         # Evaluation
   321:         if (global_step + 1) % args.eval_freq == 0:
   322:             eval_returns = eval_qnetwork(
   323:                 args.env_id, algorithm.q_network, device,
   324:                 n_episodes=args.eval_episodes, seed=args.seed + 1000,
   325:             )
   326:             mean_return = eval_returns.mean()
   327:             print(f"Eval episodic_return: {mean_return:.2f}", flush=True)
   328: 
   329:     envs.close()
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `qr_dqn` baseline — editable region  [READ-ONLY — reference implementation]

In `cleanrl/cleanrl/custom_value_discrete.py`:

```python
Lines 174–266:
   171: # =====================================================================
   172: # EDITABLE: QNetwork head and ValueAlgorithm
   173: # =====================================================================
   174: class QNetwork(nn.Module):
   175:     """Quantile Q-network for QR-DQN: MLPEncoder (fixed) + n_actions x n_quantiles head."""
   176: 
   177:     def __init__(self, obs_dim, n_actions, n_quantiles=50):
   178:         super().__init__()
   179:         self.n_actions = n_actions
   180:         self.n_quantiles = n_quantiles
   181:         self.encoder = MLPEncoder(obs_dim)
   182:         self.head = nn.Linear(ENCODER_FEATURE_DIM, n_actions * n_quantiles)
   183: 
   184:     def forward(self, obs):
   185:         """Return Q-values as mean of quantile values per action."""
   186:         features = self.encoder(obs)
   187:         quantiles = self.head(features).view(len(obs), self.n_actions, self.n_quantiles)
   188:         q_values = quantiles.mean(dim=2)
   189:         return q_values
   190: 
   191:     def get_quantiles(self, obs):
   192:         """Return raw quantile values: [batch, n_actions, n_quantiles]."""
   193:         features = self.encoder(obs)
   194:         return self.head(features).view(len(obs), self.n_actions, self.n_quantiles)
   195: 
   196: 
   197: class ValueAlgorithm:
   198:     """QR-DQN -- Quantile Regression DQN with distributional value learning."""
   199: 
   200:     def __init__(self, obs_dim, n_actions, device, args):
   201:         self.device = device
   202:         self.n_actions = n_actions
   203:         self.gamma = args.gamma
   204:         self.n_quantiles = 50
   205:         self.kappa = 1.0  # Huber loss threshold
   206:         self.total_it = 0
   207: 
   208:         self.q_network = QNetwork(obs_dim, n_actions, self.n_quantiles).to(device)
   209:         self.target_network = QNetwork(obs_dim, n_actions, self.n_quantiles).to(device)
   210:         self.target_network.load_state_dict(self.q_network.state_dict())
   211: 
   212:         self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate)
   213: 
   214:         # Fixed quantile midpoints: tau_i = (2i - 1) / (2N) for i = 1, ..., N
   215:         self.tau = torch.arange(1, self.n_quantiles + 1, dtype=torch.float32, device=device)
   216:         self.tau = (2 * self.tau - 1) / (2 * self.n_quantiles)
   217: 
   218:     def select_action(self, obs, epsilon):
   219:         if random.random() < epsilon:
   220:             return random.randint(0, self.n_actions - 1)
   221:         obs_t = torch.tensor(obs.reshape(1, -1), device=self.device, dtype=torch.float32)
   222:         q_values = self.q_network(obs_t)
   223:         return torch.argmax(q_values, dim=1).item()
   224: 
   225:     def update(self, batch, global_step):
   226:         self.total_it += 1
   227:         obs, next_obs, actions, rewards, dones = batch
   228: 
   229:         with torch.no_grad():
   230:             # Get quantile values for next state from target network
   231:             next_quantiles = self.target_network.get_quantiles(next_obs)  # [batch, n_actions, n_quantiles]
   232:             next_q = next_quantiles.mean(dim=2)  # [batch, n_actions]
   233:             next_actions = next_q.argmax(dim=1)  # [batch]
   234:             # Select quantiles for best actions
   235:             next_quantiles_best = next_quantiles[torch.arange(len(next_obs)), next_actions]  # [batch, n_quantiles]
   236:             # Compute target quantile values
   237:             target_quantiles = rewards.unsqueeze(1) + self.gamma * next_quantiles_best * (1 - dones.unsqueeze(1))
   238: 
   239:         # Get current quantile values for taken actions
   240:         current_quantiles = self.q_network.get_quantiles(obs)  # [batch, n_actions, n_quantiles]
   241:         current_quantiles = current_quantiles[torch.arange(len(obs)), actions]  # [batch, n_quantiles]
   242: 
   243:         # Quantile Huber loss
   244:         # current_quantiles: [batch, n_quantiles] (predictions at each quantile)
   245:         # target_quantiles:  [batch, n_quantiles] (targets)
   246:         # Pairwise TD errors: [batch, n_quantiles (pred), n_quantiles (target)]
   247:         td_errors = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)  # [batch, N, N]
   248: 
   249:         # Huber loss element-wise
   250:         abs_td = td_errors.abs()
   251:         huber = torch.where(abs_td <= self.kappa,
   252:                             0.5 * td_errors ** 2,
   253:                             self.kappa * (abs_td - 0.5 * self.kappa))
   254: 
   255:         # Asymmetric weighting by quantile level
   256:         # tau shape: [N] -> [1, N, 1] for broadcasting
   257:         tau = self.tau.view(1, -1, 1)
   258:         quantile_weights = torch.abs(tau - (td_errors < 0).float())
   259:         loss = (quantile_weights * huber / self.kappa).sum(dim=2).mean(dim=1).mean()
   260: 
   261:         self.optimizer.zero_grad()
   262:         loss.backward()
   263:         self.optimizer.step()
   264: 
   265:         q_values = current_quantiles.mean(dim=1)
   266:         return {"td_loss": loss.item(), "q_values": q_values.mean().item()}
   267: 
   268: 
   269: # =====================================================================
```

### `dueling_dqn` baseline — editable region  [READ-ONLY — reference implementation]

In `cleanrl/cleanrl/custom_value_discrete.py`:

```python
Lines 174–231:
   171: # =====================================================================
   172: # EDITABLE: QNetwork head and ValueAlgorithm
   173: # =====================================================================
   174: class QNetwork(nn.Module):
   175:     """Dueling Q-network: MLPEncoder (fixed) + separate value and advantage heads."""
   176: 
   177:     def __init__(self, obs_dim, n_actions):
   178:         super().__init__()
   179:         self.encoder = MLPEncoder(obs_dim)
   180:         # Value stream
   181:         self.value_head = nn.Linear(ENCODER_FEATURE_DIM, 1)
   182:         # Advantage stream
   183:         self.advantage_head = nn.Linear(ENCODER_FEATURE_DIM, n_actions)
   184: 
   185:     def forward(self, obs):
   186:         features = self.encoder(obs)
   187:         value = self.value_head(features)
   188:         advantage = self.advantage_head(features)
   189:         # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
   190:         return value + advantage - advantage.mean(dim=1, keepdim=True)
   191: 
   192: 
   193: class ValueAlgorithm:
   194:     """DuelingDQN -- Dueling Deep Q-Network."""
   195: 
   196:     def __init__(self, obs_dim, n_actions, device, args):
   197:         self.device = device
   198:         self.n_actions = n_actions
   199:         self.gamma = args.gamma
   200:         self.total_it = 0
   201: 
   202:         self.q_network = QNetwork(obs_dim, n_actions).to(device)
   203:         self.target_network = QNetwork(obs_dim, n_actions).to(device)
   204:         self.target_network.load_state_dict(self.q_network.state_dict())
   205: 
   206:         self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate)
   207: 
   208:     def select_action(self, obs, epsilon):
   209:         if random.random() < epsilon:
   210:             return random.randint(0, self.n_actions - 1)
   211:         obs_t = torch.tensor(obs.reshape(1, -1), device=self.device, dtype=torch.float32)
   212:         q_values = self.q_network(obs_t)
   213:         return torch.argmax(q_values, dim=1).item()
   214: 
   215:     def update(self, batch, global_step):
   216:         self.total_it += 1
   217:         obs, next_obs, actions, rewards, dones = batch
   218: 
   219:         with torch.no_grad():
   220:             target_max, _ = self.target_network(next_obs).max(dim=1)
   221:             td_target = rewards + (1 - dones) * self.gamma * target_max
   222: 
   223:         old_val = self.q_network(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
   224:         td_loss = F.mse_loss(td_target, old_val)
   225: 
   226:         self.optimizer.zero_grad()
   227:         td_loss.backward()
   228:         torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
   229:         self.optimizer.step()
   230: 
   231:         return {"td_loss": td_loss.item(), "q_values": old_val.mean().item()}
   232: 
   233: 
   234: # =====================================================================
```

### `c51` baseline — editable region  [READ-ONLY — reference implementation]

In `cleanrl/cleanrl/custom_value_discrete.py`:

```python
Lines 174–255:
   171: # =====================================================================
   172: # EDITABLE: QNetwork head and ValueAlgorithm
   173: # =====================================================================
   174: class QNetwork(nn.Module):
   175:     """Distributional Q-network for C51: MLPEncoder (fixed) + n_actions x n_atoms head."""
   176: 
   177:     def __init__(self, obs_dim, n_actions, n_atoms=51, v_min=-500, v_max=500):
   178:         super().__init__()
   179:         self.n_actions = n_actions
   180:         self.n_atoms = n_atoms
   181:         self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=n_atoms))
   182:         self.encoder = MLPEncoder(obs_dim)
   183:         self.head = nn.Linear(ENCODER_FEATURE_DIM, n_actions * n_atoms)
   184: 
   185:     def forward(self, obs):
   186:         features = self.encoder(obs)
   187:         logits = self.head(features)
   188:         pmfs = torch.softmax(logits.view(len(obs), self.n_actions, self.n_atoms), dim=2)
   189:         q_values = (pmfs * self.atoms).sum(2)
   190:         return q_values
   191: 
   192:     def get_action(self, obs, action=None):
   193:         features = self.encoder(obs)
   194:         logits = self.head(features)
   195:         pmfs = torch.softmax(logits.view(len(obs), self.n_actions, self.n_atoms), dim=2)
   196:         q_values = (pmfs * self.atoms).sum(2)
   197:         if action is None:
   198:             action = torch.argmax(q_values, 1)
   199:         return action, pmfs[torch.arange(len(obs)), action]
   200: 
   201: 
   202: class ValueAlgorithm:
   203:     """C51 -- Categorical DQN with distributional value learning."""
   204: 
   205:     def __init__(self, obs_dim, n_actions, device, args):
   206:         self.device = device
   207:         self.n_actions = n_actions
   208:         self.gamma = args.gamma
   209:         self.n_atoms = 51
   210:         self.v_min = -500.0
   211:         self.v_max = 500.0
   212:         self.total_it = 0
   213: 
   214:         self.q_network = QNetwork(obs_dim, n_actions, self.n_atoms, self.v_min, self.v_max).to(device)
   215:         self.target_network = QNetwork(obs_dim, n_actions, self.n_atoms, self.v_min, self.v_max).to(device)
   216:         self.target_network.load_state_dict(self.q_network.state_dict())
   217: 
   218:         self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate)
   219: 
   220:     def select_action(self, obs, epsilon):
   221:         if random.random() < epsilon:
   222:             return random.randint(0, self.n_actions - 1)
   223:         obs_t = torch.tensor(obs.reshape(1, -1), device=self.device, dtype=torch.float32)
   224:         action, _ = self.q_network.get_action(obs_t)
   225:         return action.item()
   226: 
   227:     def update(self, batch, global_step):
   228:         self.total_it += 1
   229:         obs, next_obs, actions, rewards, dones = batch
   230: 
   231:         with torch.no_grad():
   232:             _, next_pmfs = self.target_network.get_action(next_obs)
   233:             next_atoms = rewards.unsqueeze(1) + self.gamma * self.target_network.atoms * (1 - dones.unsqueeze(1))
   234:             # Projection
   235:             delta_z = self.target_network.atoms[1] - self.target_network.atoms[0]
   236:             tz = next_atoms.clamp(self.v_min, self.v_max)
   237:             b = (tz - self.v_min) / delta_z
   238:             l = b.floor().clamp(0, self.n_atoms - 1)
   239:             u = b.ceil().clamp(0, self.n_atoms - 1)
   240:             d_m_l = (u + (l == u).float() - b) * next_pmfs
   241:             d_m_u = (b - l) * next_pmfs
   242:             target_pmfs = torch.zeros_like(next_pmfs)
   243:             for i in range(target_pmfs.size(0)):
   244:                 target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
   245:                 target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])
   246: 
   247:         _, old_pmfs = self.q_network.get_action(obs, actions)
   248:         loss = (-(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean()
   249: 
   250:         self.optimizer.zero_grad()
   251:         loss.backward()
   252:         self.optimizer.step()
   253: 
   254:         q_values = (old_pmfs * self.q_network.atoms).sum(1)
   255:         return {"td_loss": loss.item(), "q_values": q_values.mean().item()}
   256: 
   257: 
   258: # =====================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
