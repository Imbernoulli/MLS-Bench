# MLS-Bench: rl-offpolicy-continuous

# Online RL: Off-Policy Actor-Critic for Continuous Control

## Research Question
Design and implement an off-policy actor-critic RL algorithm for
continuous control. Your code goes in `custom_offpolicy_continuous.py`.
Several reference implementations are provided as read-only `*.edit.py`
baselines.

## Background
Off-policy methods maintain a replay buffer of past transitions and
update the policy using data collected under previous policies. They are
typically more sample-efficient than on-policy methods, but they expose
well-known failure modes: overestimation bias in Q-value estimates,
instability of the actor under noisy critic targets, and
exploration-exploitation tradeoffs. Different design points address
these via twin / ensemble critics, target smoothing, entropy
regularization, delayed updates, or batch-normalized critics.

Reference baselines spanning the design space:
- **DDPG** — Lillicrap et al., "Continuous Control with Deep
  Reinforcement Learning" (arXiv:1509.02971, ICLR 2016). Single
  deterministic actor and critic with target networks and
  Ornstein-Uhlenbeck (or Gaussian) exploration noise.
- **TD3** — Fujimoto et al., "Addressing Function Approximation Error in
  Actor-Critic Methods" (arXiv:1802.09477, ICML 2018). Twin critics with
  clipped-double-Q targets, target-policy smoothing, and delayed actor
  updates (default policy delay `d = 2`).
- **SAC** — Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum
  Entropy Deep Reinforcement Learning with a Stochastic Actor"
  (arXiv:1801.01290, ICML 2018). Stochastic actor with maximum-entropy
  objective and twin-Q targets.

## Constraints
- Network architecture dimensions are FIXED and cannot be modified.
- Total parameter count is enforced at runtime; the contribution must be
  algorithmic (losses, target construction, exploration, update rules)
  rather than capacity.
- Do **not** simply copy a reference implementation with minor changes.

## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/cleanrl/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `cleanrl/cleanrl/custom_offpolicy_continuous.py`
- editable lines **153–244**




## Readable Context


### `cleanrl/cleanrl/custom_offpolicy_continuous.py`  [EDITABLE — lines 153–244 only]

```python
     1: # Custom off-policy continuous RL algorithm for MLS-Bench
     2: #
     3: # EDITABLE section: Actor, QNetwork, and OffPolicyAlgorithm classes.
     4: # FIXED sections: everything else (config, env, buffer, eval, training loop).
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
    34:     env_id: str = "HalfCheetah-v4"
    35:     """the id of the environment"""
    36:     total_timesteps: int = 1000000
    37:     """total timesteps of the experiments"""
    38:     learning_rate: float = 3e-4
    39:     """the learning rate of the optimizer"""
    40:     buffer_size: int = int(1e6)
    41:     """the replay memory buffer size"""
    42:     gamma: float = 0.99
    43:     """the discount factor gamma"""
    44:     tau: float = 0.005
    45:     """target smoothing coefficient (default: 0.005)"""
    46:     batch_size: int = 256
    47:     """the batch size of sample from the replay memory"""
    48:     learning_starts: int = 25000
    49:     """timestep to start learning"""
    50:     policy_frequency: int = 2
    51:     """the frequency of training policy (delayed)"""
    52:     exploration_noise: float = 0.1
    53:     """the scale of exploration noise"""
    54:     eval_freq: int = 10000
    55:     """evaluation frequency (timesteps)"""
    56:     eval_episodes: int = 10
    57:     """number of evaluation episodes"""
    58: 
    59: 
    60: # =====================================================================
    61: # FIXED: Environment setup
    62: # =====================================================================
    63: def make_env(env_id, seed):
    64:     def thunk():
    65:         env = gym.make(env_id)
    66:         env = gym.wrappers.RecordEpisodeStatistics(env)
    67:         env.action_space.seed(seed)
    68:         return env
    69:     return thunk
    70: 
    71: 
    72: # =====================================================================
    73: # FIXED: Replay Buffer
    74: # =====================================================================
    75: class SimpleReplayBuffer:
    76:     """Numpy-based replay buffer for continuous actions."""
    77: 
    78:     def __init__(self, obs_dim, action_dim, max_size=int(1e6)):
    79:         self.max_size = max_size
    80:         self.ptr = 0
    81:         self.size = 0
    82:         self.obs = np.zeros((max_size, obs_dim), dtype=np.float32)
    83:         self.next_obs = np.zeros((max_size, obs_dim), dtype=np.float32)
    84:         self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
    85:         self.rewards = np.zeros((max_size,), dtype=np.float32)
    86:         self.dones = np.zeros((max_size,), dtype=np.float32)
    87: 
    88:     def add(self, obs, next_obs, action, reward, done):
    89:         self.obs[self.ptr] = obs
    90:         self.next_obs[self.ptr] = next_obs
    91:         self.actions[self.ptr] = action
    92:         self.rewards[self.ptr] = reward
    93:         self.dones[self.ptr] = done
    94:         self.ptr = (self.ptr + 1) % self.max_size
    95:         self.size = min(self.size + 1, self.max_size)
    96: 
    97:     def sample(self, batch_size, device):
    98:         idx = np.random.randint(0, self.size, size=batch_size)
    99:         return (
   100:             torch.tensor(self.obs[idx], device=device),
   101:             torch.tensor(self.next_obs[idx], device=device),
   102:             torch.tensor(self.actions[idx], device=device),
   103:             torch.tensor(self.rewards[idx], device=device),
   104:             torch.tensor(self.dones[idx], device=device),
   105:         )
   106: 
   107: 
   108: # =====================================================================
   109: # FIXED: Utilities
   110: # =====================================================================
   111: def soft_update(target, source, tau):
   112:     for tp, sp in zip(target.parameters(), source.parameters()):
   113:         tp.data.copy_((1 - tau) * tp.data + tau * sp.data)
   114: 
   115: 
   116: def _mlp_factory(input_dim, output_dim, hidden=256):
   117:     """Build a 2-hidden-layer MLP. Use this as a building block for actors/critics."""
   118:     return nn.Sequential(
   119:         nn.Linear(input_dim, hidden),
   120:         nn.ReLU(),
   121:         nn.Linear(hidden, hidden),
   122:         nn.ReLU(),
   123:         nn.Linear(hidden, output_dim),
   124:     )
   125: 
   126: 
   127: @torch.no_grad()
   128: def eval_actor(env_id, actor, device, n_episodes, seed):
   129:     """Evaluate actor for n_episodes in a fresh env; returns array of episode rewards."""
   130:     eval_env = gym.make(env_id)
   131:     episode_rewards = []
   132:     for ep in range(n_episodes):
   133:         obs, _ = eval_env.reset(seed=seed + ep)
   134:         done = False
   135:         episode_reward = 0.0
   136:         while not done:
   137:             obs_t = torch.tensor(obs.reshape(1, -1), device=device, dtype=torch.float32)
   138:             action = actor.get_action(obs_t)
   139:             if isinstance(action, tuple):
   140:                 action = action[0]
   141:             action = action.cpu().numpy().flatten()
   142:             obs, reward, terminated, truncated, _ = eval_env.step(action)
   143:             done = terminated or truncated
   144:             episode_reward += reward
   145:         episode_rewards.append(episode_reward)
   146:     eval_env.close()
   147:     return np.asarray(episode_rewards)
   148: 
   149: 
   150: # =====================================================================
   151: # EDITABLE: Network definitions and OffPolicyAlgorithm
   152: # =====================================================================
   153: class Actor(nn.Module):
   154:     """Actor network. Must implement forward(obs) and get_action(obs).
   155: 
   156:     forward(obs) -> action tensor (used for training).
   157:     get_action(obs) -> action tensor (used for evaluation, no grad).
   158:     """
   159: 
   160:     def __init__(self, obs_dim, action_dim, max_action):
   161:         super().__init__()
   162:         self.max_action = max_action
   163:         self.fc1 = nn.Linear(obs_dim, 256)
   164:         self.fc2 = nn.Linear(256, 256)
   165:         self.fc_mu = nn.Linear(256, action_dim)
   166:         self.register_buffer("action_scale", torch.tensor(max_action, dtype=torch.float32))
   167: 
   168:     def forward(self, obs):
   169:         x = F.relu(self.fc1(obs))
   170:         x = F.relu(self.fc2(x))
   171:         return torch.tanh(self.fc_mu(x)) * self.action_scale
   172: 
   173:     @torch.no_grad()
   174:     def get_action(self, obs):
   175:         return self.forward(obs)
   176: 
   177: 
   178: class QNetwork(nn.Module):
   179:     """Q-function Q(s, a) -> scalar."""
   180: 
   181:     def __init__(self, obs_dim, action_dim):
   182:         super().__init__()
   183:         self.fc1 = nn.Linear(obs_dim + action_dim, 256)
   184:         self.fc2 = nn.Linear(256, 256)
   185:         self.fc3 = nn.Linear(256, 1)
   186: 
   187:     def forward(self, obs, action):
   188:         x = torch.cat([obs, action], dim=-1)
   189:         x = F.relu(self.fc1(x))
   190:         x = F.relu(self.fc2(x))
   191:         return self.fc3(x)
   192: 
   193: 
   194: class OffPolicyAlgorithm:
   195:     """Off-policy actor-critic algorithm -- implement your approach here.
   196: 
   197:     The training loop calls:
   198:         algorithm = OffPolicyAlgorithm(obs_dim, action_dim, max_action, device, args)
   199:         action = algorithm.select_action(obs)        # during data collection
   200:         metrics = algorithm.update(batch)             # after each env step
   201:         eval_actor(env_id, algorithm.actor, ...)      # periodic evaluation
   202: 
   203:     You MUST set self.actor to an nn.Module with a .get_action(obs) method.
   204: 
   205:     Available classes (defined above, editable):
   206:         Actor    -- deterministic policy with tanh squashing
   207:         QNetwork -- Q(s, a) critic
   208: 
   209:     Available utilities (fixed): soft_update, _mlp_factory
   210:     """
   211: 
   212:     def __init__(self, obs_dim, action_dim, max_action, device, args):
   213:         self.device = device
   214:         self.max_action = max_action
   215:         self.gamma = args.gamma
   216:         self.tau = args.tau
   217:         self.total_it = 0
   218: 
   219:         # Build networks -- modify or replace as needed
   220:         self.actor = Actor(obs_dim, action_dim, max_action).to(device)
   221:         self.qf1 = QNetwork(obs_dim, action_dim).to(device)
   222: 
   223:         self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.learning_rate)
   224:         self.q_optimizer = optim.Adam(self.qf1.parameters(), lr=args.learning_rate)
   225: 
   226:     def select_action(self, obs):
   227:         """Select action for environment interaction (with exploration noise)."""
   228:         obs_t = torch.tensor(obs.reshape(1, -1), device=self.device, dtype=torch.float32)
   229:         action = self.actor(obs_t).cpu().numpy().flatten()
   230:         noise = np.random.normal(0, self.max_action * 0.1, size=action.shape)
   231:         return np.clip(action + noise, -self.max_action, self.max_action)
   232: 
   233:     def update(self, batch):
   234:         """Single gradient update. Returns a dict of scalar metrics.
   235: 
   236:         batch = (obs, next_obs, actions, rewards, dones) -- torch.Tensor on device
   237: 
   238:         TODO: implement your off-policy RL algorithm here.
   239:         """
   240:         self.total_it += 1
   241:         obs, next_obs, actions, rewards, dones = batch
   242: 
   243:         # Placeholder -- replace with your algorithm
   244:         return {"critic_loss": 0.0, "actor_loss": 0.0}
   245: 
   246: 
   247: # =====================================================================
   248: # FIXED: Training loop
   249: # =====================================================================
   250: if __name__ == "__main__":
   251:     args = tyro.cli(Args)
   252:     run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
   253: 
   254:     # Seeding
   255:     random.seed(args.seed)
   256:     np.random.seed(args.seed)
   257:     torch.manual_seed(args.seed)
   258:     torch.backends.cudnn.deterministic = args.torch_deterministic
   259: 
   260:     device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
   261: 
   262:     # Environment setup
   263:     envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed)])
   264:     assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
   265: 
   266:     obs_dim = np.array(envs.single_observation_space.shape).prod()
   267:     action_dim = np.prod(envs.single_action_space.shape)
   268:     max_action = float(envs.single_action_space.high[0])
   269: 
   270:     # Algorithm
   271:     algorithm = OffPolicyAlgorithm(obs_dim, action_dim, max_action, device, args)
   272: 
   273:     # Replay buffer
   274:     rb = SimpleReplayBuffer(obs_dim, action_dim, args.buffer_size)
   275: 
   276:     start_time = time.time()
   277:     obs, _ = envs.reset(seed=args.seed)
   278: 
   279:     for global_step in range(args.total_timesteps):
   280:         # Action selection
   281:         if global_step < args.learning_starts:
   282:             actions = np.array([envs.single_action_space.sample()])
   283:         else:
   284:             actions = algorithm.select_action(obs[0]).reshape(1, -1)
   285: 
   286:         # Environment step
   287:         next_obs, rewards, terminations, truncations, infos = envs.step(actions)
   288: 
   289:         if "final_info" in infos:
   290:             for info in infos["final_info"]:
   291:                 if info is not None:
   292:                     print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
   293:                     break
   294: 
   295:         # Handle truncation
   296:         real_next_obs = next_obs.copy()
   297:         for idx, trunc in enumerate(truncations):
   298:             if trunc:
   299:                 real_next_obs[idx] = infos["final_observation"][idx]
   300: 
   301:         rb.add(obs[0], real_next_obs[0], actions[0], rewards[0], terminations[0])
   302:         obs = next_obs
   303: 
   304:         # Training
   305:         if global_step > args.learning_starts:
   306:             batch = rb.sample(args.batch_size, device)
   307:             log_dict = algorithm.update(batch)
   308: 
   309:             if global_step % 1000 == 0:
   310:                 metrics_str = " ".join(
   311:                     f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
   312:                     for k, v in log_dict.items()
   313:                 )
   314:                 print(f"TRAIN_METRICS step={global_step} {metrics_str}", flush=True)
   315: 
   316:         # Evaluation
   317:         if (global_step + 1) % args.eval_freq == 0:
   318:             eval_returns = eval_actor(
   319:                 args.env_id, algorithm.actor, device,
   320:                 n_episodes=args.eval_episodes, seed=args.seed + 1000,
   321:             )
   322:             mean_return = eval_returns.mean()
   323:             print(f"Eval episodic_return: {mean_return:.2f}", flush=True)
   324: 
   325:     envs.close()
```

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


### `ddpg` baseline — editable region  [READ-ONLY — reference implementation]

In `cleanrl/cleanrl/custom_offpolicy_continuous.py`:

```python
Lines 153–251:
   150: # =====================================================================
   151: # EDITABLE: Network definitions and OffPolicyAlgorithm
   152: # =====================================================================
   153: class Actor(nn.Module):
   154:     """Actor network. Must implement forward(obs) and get_action(obs).
   155: 
   156:     forward(obs) -> action tensor (used for training).
   157:     get_action(obs) -> action tensor (used for evaluation, no grad).
   158:     """
   159: 
   160:     def __init__(self, obs_dim, action_dim, max_action):
   161:         super().__init__()
   162:         self.max_action = max_action
   163:         self.fc1 = nn.Linear(obs_dim, 256)
   164:         self.fc2 = nn.Linear(256, 256)
   165:         self.fc_mu = nn.Linear(256, action_dim)
   166:         self.register_buffer("action_scale", torch.tensor(max_action, dtype=torch.float32))
   167: 
   168:     def forward(self, obs):
   169:         x = F.relu(self.fc1(obs))
   170:         x = F.relu(self.fc2(x))
   171:         return torch.tanh(self.fc_mu(x)) * self.action_scale
   172: 
   173:     @torch.no_grad()
   174:     def get_action(self, obs):
   175:         return self.forward(obs)
   176: 
   177: 
   178: class QNetwork(nn.Module):
   179:     """Q-function Q(s, a) -> scalar."""
   180: 
   181:     def __init__(self, obs_dim, action_dim):
   182:         super().__init__()
   183:         self.fc1 = nn.Linear(obs_dim + action_dim, 256)
   184:         self.fc2 = nn.Linear(256, 256)
   185:         self.fc3 = nn.Linear(256, 1)
   186: 
   187:     def forward(self, obs, action):
   188:         x = torch.cat([obs, action], dim=-1)
   189:         x = F.relu(self.fc1(x))
   190:         x = F.relu(self.fc2(x))
   191:         return self.fc3(x)
   192: 
   193: 
   194: class OffPolicyAlgorithm:
   195:     """DDPG — Deep Deterministic Policy Gradient."""
   196: 
   197:     def __init__(self, obs_dim, action_dim, max_action, device, args):
   198:         self.device = device
   199:         self.max_action = max_action
   200:         self.gamma = args.gamma
   201:         self.tau = args.tau
   202:         self.exploration_noise = args.exploration_noise
   203:         self.policy_frequency = args.policy_frequency
   204:         self.total_it = 0
   205: 
   206:         self.actor = Actor(obs_dim, action_dim, max_action).to(device)
   207:         self.target_actor = Actor(obs_dim, action_dim, max_action).to(device)
   208:         self.target_actor.load_state_dict(self.actor.state_dict())
   209: 
   210:         self.qf1 = QNetwork(obs_dim, action_dim).to(device)
   211:         self.qf1_target = QNetwork(obs_dim, action_dim).to(device)
   212:         self.qf1_target.load_state_dict(self.qf1.state_dict())
   213: 
   214:         self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.learning_rate)
   215:         self.q_optimizer = optim.Adam(self.qf1.parameters(), lr=args.learning_rate)
   216: 
   217:     def select_action(self, obs):
   218:         obs_t = torch.tensor(obs.reshape(1, -1), device=self.device, dtype=torch.float32)
   219:         with torch.no_grad():
   220:             action = self.actor(obs_t).cpu().numpy().flatten()
   221:         noise = np.random.normal(0, self.max_action * self.exploration_noise, size=action.shape)
   222:         return np.clip(action + noise, -self.max_action, self.max_action)
   223: 
   224:     def update(self, batch):
   225:         self.total_it += 1
   226:         obs, next_obs, actions, rewards, dones = batch
   227: 
   228:         with torch.no_grad():
   229:             next_actions = self.target_actor(next_obs)
   230:             target_q = self.qf1_target(next_obs, next_actions).view(-1)
   231:             td_target = rewards + (1 - dones) * self.gamma * target_q
   232: 
   233:         current_q = self.qf1(obs, actions).view(-1)
   234:         critic_loss = F.mse_loss(current_q, td_target)
   235: 
   236:         self.q_optimizer.zero_grad()
   237:         critic_loss.backward()
   238:         self.q_optimizer.step()
   239: 
   240:         actor_loss_val = 0.0
   241:         if self.total_it % self.policy_frequency == 0:
   242:             actor_loss = -self.qf1(obs, self.actor(obs)).mean()
   243:             self.actor_optimizer.zero_grad()
   244:             actor_loss.backward()
   245:             self.actor_optimizer.step()
   246:             actor_loss_val = actor_loss.item()
   247: 
   248:             soft_update(self.target_actor, self.actor, self.tau)
   249:             soft_update(self.qf1_target, self.qf1, self.tau)
   250: 
   251:         return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss_val}
   252: 
   253: 
   254: # =====================================================================
```

### `td3` baseline — editable region  [READ-ONLY — reference implementation]

In `cleanrl/cleanrl/custom_offpolicy_continuous.py`:

```python
Lines 153–267:
   150: # =====================================================================
   151: # EDITABLE: Network definitions and OffPolicyAlgorithm
   152: # =====================================================================
   153: class Actor(nn.Module):
   154:     """Actor network. Must implement forward(obs) and get_action(obs).
   155: 
   156:     forward(obs) -> action tensor (used for training).
   157:     get_action(obs) -> action tensor (used for evaluation, no grad).
   158:     """
   159: 
   160:     def __init__(self, obs_dim, action_dim, max_action):
   161:         super().__init__()
   162:         self.max_action = max_action
   163:         self.fc1 = nn.Linear(obs_dim, 256)
   164:         self.fc2 = nn.Linear(256, 256)
   165:         self.fc_mu = nn.Linear(256, action_dim)
   166:         self.register_buffer("action_scale", torch.tensor(max_action, dtype=torch.float32))
   167: 
   168:     def forward(self, obs):
   169:         x = F.relu(self.fc1(obs))
   170:         x = F.relu(self.fc2(x))
   171:         return torch.tanh(self.fc_mu(x)) * self.action_scale
   172: 
   173:     @torch.no_grad()
   174:     def get_action(self, obs):
   175:         return self.forward(obs)
   176: 
   177: 
   178: class QNetwork(nn.Module):
   179:     """Q-function Q(s, a) -> scalar."""
   180: 
   181:     def __init__(self, obs_dim, action_dim):
   182:         super().__init__()
   183:         self.fc1 = nn.Linear(obs_dim + action_dim, 256)
   184:         self.fc2 = nn.Linear(256, 256)
   185:         self.fc3 = nn.Linear(256, 1)
   186: 
   187:     def forward(self, obs, action):
   188:         x = torch.cat([obs, action], dim=-1)
   189:         x = F.relu(self.fc1(x))
   190:         x = F.relu(self.fc2(x))
   191:         return self.fc3(x)
   192: 
   193: 
   194: class OffPolicyAlgorithm:
   195:     """TD3 — Twin Delayed Deep Deterministic Policy Gradient."""
   196: 
   197:     def __init__(self, obs_dim, action_dim, max_action, device, args):
   198:         self.device = device
   199:         self.max_action = max_action
   200:         self.gamma = args.gamma
   201:         self.tau = args.tau
   202:         self.exploration_noise = args.exploration_noise
   203:         self.policy_frequency = args.policy_frequency
   204:         self.policy_noise = 0.2
   205:         self.noise_clip = 0.5
   206:         self.total_it = 0
   207: 
   208:         self.actor = Actor(obs_dim, action_dim, max_action).to(device)
   209:         self.target_actor = Actor(obs_dim, action_dim, max_action).to(device)
   210:         self.target_actor.load_state_dict(self.actor.state_dict())
   211: 
   212:         self.qf1 = QNetwork(obs_dim, action_dim).to(device)
   213:         self.qf2 = QNetwork(obs_dim, action_dim).to(device)
   214:         self.qf1_target = QNetwork(obs_dim, action_dim).to(device)
   215:         self.qf2_target = QNetwork(obs_dim, action_dim).to(device)
   216:         self.qf1_target.load_state_dict(self.qf1.state_dict())
   217:         self.qf2_target.load_state_dict(self.qf2.state_dict())
   218: 
   219:         self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.learning_rate)
   220:         self.q_optimizer = optim.Adam(
   221:             list(self.qf1.parameters()) + list(self.qf2.parameters()),
   222:             lr=args.learning_rate,
   223:         )
   224: 
   225:     def select_action(self, obs):
   226:         obs_t = torch.tensor(obs.reshape(1, -1), device=self.device, dtype=torch.float32)
   227:         with torch.no_grad():
   228:             action = self.actor(obs_t).cpu().numpy().flatten()
   229:         noise = np.random.normal(0, self.max_action * self.exploration_noise, size=action.shape)
   230:         return np.clip(action + noise, -self.max_action, self.max_action)
   231: 
   232:     def update(self, batch):
   233:         self.total_it += 1
   234:         obs, next_obs, actions, rewards, dones = batch
   235: 
   236:         with torch.no_grad():
   237:             noise = (torch.randn_like(actions) * self.policy_noise).clamp(
   238:                 -self.noise_clip, self.noise_clip
   239:             ) * self.max_action
   240:             next_actions = (self.target_actor(next_obs) + noise).clamp(
   241:                 -self.max_action, self.max_action
   242:             )
   243:             target_q1 = self.qf1_target(next_obs, next_actions).view(-1)
   244:             target_q2 = self.qf2_target(next_obs, next_actions).view(-1)
   245:             td_target = rewards + (1 - dones) * self.gamma * torch.min(target_q1, target_q2)
   246: 
   247:         q1 = self.qf1(obs, actions).view(-1)
   248:         q2 = self.qf2(obs, actions).view(-1)
   249:         critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)
   250: 
   251:         self.q_optimizer.zero_grad()
   252:         critic_loss.backward()
   253:         self.q_optimizer.step()
   254: 
   255:         actor_loss_val = 0.0
   256:         if self.total_it % self.policy_frequency == 0:
   257:             actor_loss = -self.qf1(obs, self.actor(obs)).mean()
   258:             self.actor_optimizer.zero_grad()
   259:             actor_loss.backward()
   260:             self.actor_optimizer.step()
   261:             actor_loss_val = actor_loss.item()
   262: 
   263:             soft_update(self.target_actor, self.actor, self.tau)
   264:             soft_update(self.qf1_target, self.qf1, self.tau)
   265:             soft_update(self.qf2_target, self.qf2, self.tau)
   266: 
   267:         return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss_val}
   268: 
   269: 
   270: # =====================================================================
```

### `sac` baseline — editable region  [READ-ONLY — reference implementation]

In `cleanrl/cleanrl/custom_offpolicy_continuous.py`:

```python
Lines 153–292:
   150: # =====================================================================
   151: # EDITABLE: Network definitions and OffPolicyAlgorithm
   152: # =====================================================================
   153: LOG_STD_MAX = 2
   154: LOG_STD_MIN = -5
   155: 
   156: 
   157: class Actor(nn.Module):
   158:     """Stochastic Tanh-Gaussian actor for SAC."""
   159: 
   160:     def __init__(self, obs_dim, action_dim, max_action):
   161:         super().__init__()
   162:         self.max_action = max_action
   163:         self.fc1 = nn.Linear(obs_dim, 256)
   164:         self.fc2 = nn.Linear(256, 256)
   165:         self.fc_mean = nn.Linear(256, action_dim)
   166:         self.fc_logstd = nn.Linear(256, action_dim)
   167:         self.register_buffer("action_scale", torch.tensor(max_action, dtype=torch.float32))
   168: 
   169:     def forward(self, obs):
   170:         x = F.relu(self.fc1(obs))
   171:         x = F.relu(self.fc2(x))
   172:         mean = self.fc_mean(x)
   173:         log_std = self.fc_logstd(x)
   174:         log_std = torch.tanh(log_std)
   175:         log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
   176:         return mean, log_std
   177: 
   178:     def get_action(self, obs):
   179:         mean, log_std = self(obs)
   180:         std = log_std.exp()
   181:         normal = torch.distributions.Normal(mean, std)
   182:         x_t = normal.rsample()
   183:         y_t = torch.tanh(x_t)
   184:         action = y_t * self.action_scale
   185:         log_prob = normal.log_prob(x_t)
   186:         log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
   187:         log_prob = log_prob.sum(1, keepdim=True)
   188:         mean_action = torch.tanh(mean) * self.action_scale
   189:         return action, log_prob, mean_action
   190: 
   191: class QNetwork(nn.Module):
   192:     """Q-function Q(s, a) -> scalar."""
   193: 
   194:     def __init__(self, obs_dim, action_dim):
   195:         super().__init__()
   196:         self.fc1 = nn.Linear(obs_dim + action_dim, 256)
   197:         self.fc2 = nn.Linear(256, 256)
   198:         self.fc3 = nn.Linear(256, 1)
   199: 
   200:     def forward(self, obs, action):
   201:         x = torch.cat([obs, action], dim=-1)
   202:         x = F.relu(self.fc1(x))
   203:         x = F.relu(self.fc2(x))
   204:         return self.fc3(x)
   205: 
   206: 
   207: class OffPolicyAlgorithm:
   208:     """SAC — Soft Actor-Critic with automatic entropy tuning."""
   209: 
   210:     def __init__(self, obs_dim, action_dim, max_action, device, args):
   211:         self.device = device
   212:         self.max_action = max_action
   213:         self.gamma = args.gamma
   214:         self.tau = args.tau
   215:         self.policy_frequency = args.policy_frequency
   216:         self.total_it = 0
   217: 
   218:         self.actor = Actor(obs_dim, action_dim, max_action).to(device)
   219: 
   220:         self.qf1 = QNetwork(obs_dim, action_dim).to(device)
   221:         self.qf2 = QNetwork(obs_dim, action_dim).to(device)
   222:         self.qf1_target = QNetwork(obs_dim, action_dim).to(device)
   223:         self.qf2_target = QNetwork(obs_dim, action_dim).to(device)
   224:         self.qf1_target.load_state_dict(self.qf1.state_dict())
   225:         self.qf2_target.load_state_dict(self.qf2.state_dict())
   226: 
   227:         self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.learning_rate)
   228:         self.q_optimizer = optim.Adam(
   229:             list(self.qf1.parameters()) + list(self.qf2.parameters()),
   230:             lr=args.learning_rate,
   231:         )
   232: 
   233:         # Auto entropy tuning
   234:         self.target_entropy = -action_dim
   235:         self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
   236:         self.alpha = self.log_alpha.exp().item()
   237:         self.alpha_optimizer = optim.Adam([self.log_alpha], lr=args.learning_rate)
   238: 
   239:     def select_action(self, obs):
   240:         obs_t = torch.tensor(obs.reshape(1, -1), device=self.device, dtype=torch.float32)
   241:         with torch.no_grad():
   242:             action, _, _ = self.actor.get_action(obs_t)
   243:         return action.cpu().numpy().flatten()
   244: 
   245:     def update(self, batch):
   246:         self.total_it += 1
   247:         obs, next_obs, actions, rewards, dones = batch
   248: 
   249:         # Update critics
   250:         with torch.no_grad():
   251:             next_actions, next_log_pi, _ = self.actor.get_action(next_obs)
   252:             q1_next = self.qf1_target(next_obs, next_actions).view(-1)
   253:             q2_next = self.qf2_target(next_obs, next_actions).view(-1)
   254:             min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_pi.view(-1)
   255:             td_target = rewards + (1 - dones) * self.gamma * min_q_next
   256: 
   257:         q1 = self.qf1(obs, actions).view(-1)
   258:         q2 = self.qf2(obs, actions).view(-1)
   259:         critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)
   260: 
   261:         self.q_optimizer.zero_grad()
   262:         critic_loss.backward()
   263:         self.q_optimizer.step()
   264: 
   265:         # Update actor
   266:         actor_loss_val = 0.0
   267:         if self.total_it % self.policy_frequency == 0:
   268:             pi, log_pi, _ = self.actor.get_action(obs)
   269:             q1_pi = self.qf1(obs, pi).view(-1)
   270:             q2_pi = self.qf2(obs, pi).view(-1)
   271:             min_q_pi = torch.min(q1_pi, q2_pi)
   272:             actor_loss = (self.alpha * log_pi.view(-1) - min_q_pi).mean()
   273: 
   274:             self.actor_optimizer.zero_grad()
   275:             actor_loss.backward()
   276:             self.actor_optimizer.step()
   277:             actor_loss_val = actor_loss.item()
   278: 
   279:             # Update alpha
   280:             with torch.no_grad():
   281:                 _, log_pi_alpha, _ = self.actor.get_action(obs)
   282:             alpha_loss = (-self.log_alpha.exp() * (log_pi_alpha.view(-1) + self.target_entropy)).mean()
   283:             self.alpha_optimizer.zero_grad()
   284:             alpha_loss.backward()
   285:             self.alpha_optimizer.step()
   286:             self.alpha = self.log_alpha.exp().item()
   287: 
   288:         # Update target networks
   289:         soft_update(self.qf1_target, self.qf1, self.tau)
   290:         soft_update(self.qf2_target, self.qf2, self.tau)
   291: 
   292:         return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss_val, "alpha": self.alpha}
   293: 
   294: 
   295: # =====================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
