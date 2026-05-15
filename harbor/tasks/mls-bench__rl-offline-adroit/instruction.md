# MLS-Bench: rl-offline-adroit

# Offline RL: Dexterous Manipulation with Narrow Expert Data (Adroit)

## Research Question
Design and implement an offline RL algorithm for high-dimensional
dexterous manipulation from narrow human-demonstration data. Your code
goes in `custom_adroit.py`. Several reference implementations are
provided as read-only `*.edit.py` baselines.

## Background
Adroit tasks involve a 24-DoF simulated robotic hand with high-dimensional
action spaces (24–30 dims). The D4RL `human-v1` datasets contain only
roughly 25 human teleoperation demonstrations per task, creating severe
distribution shift compared to locomotion-style offline RL. Standard
Q-learning on this data tends to extrapolate badly outside the narrow
data support, while pure behavior cloning is limited by the small
dataset.

Reference baselines spanning the design space:
- **IQL** — Kostrikov et al., "Offline Reinforcement Learning with
  Implicit Q-Learning" (arXiv:2110.06169, ICLR 2022). Expectile
  regression with advantage-weighted policy extraction; well-suited to
  narrow data support without requiring OOD action queries.
- **AWAC** — Nair et al., "AWAC: Accelerating Online Reinforcement
  Learning with Offline Datasets" (arXiv:2006.09359). Advantage-weighted
  actor-critic with implicit policy constraint (default temperature
  `lambda = 1.0` per paper).
- **ReBRAC** — Tarasov et al., "Revisiting the Minimalist Approach to
  Offline Reinforcement Learning" (arXiv:2305.09836, NeurIPS 2023).
  TD3+BC-style actor-critic with decoupled actor / critic BC penalties;
  per-domain BC coefficients are tuned per the paper's appendix.

## Constraints
- **Network dimensions are fixed at 256.** All MLP hidden layers must
  use 256 units. A `_mlp()` factory function is provided in the FIXED
  section for convenience. You may define custom network classes but
  hidden widths must remain 256.
- **Total parameter count is enforced.** The training loop checks that
  total trainable parameters do not exceed 1.2x the largest baseline
  architecture, so the contribution must be algorithmic (loss,
  regularization, target construction, training procedure) rather than
  capacity.
- Do **not** simply copy a reference implementation with minor changes.

## Evaluation
Trained and evaluated on Adroit tasks Pen (rotation), Door (opening) and
Hammer (nailing) using the D4RL `human-v1` datasets. Metric: D4RL
normalized score (0 = random performance, 100 = expert), averaged over
evaluation rollouts. Higher is better. Strong methods should work across
the manipulation tasks rather than overfitting to a single dataset.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/CORL/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to score zero.

- `CORL/algorithms/offline/custom_adroit.py`
- editable lines **214–416**




## Readable Context


### `CORL/algorithms/offline/custom_adroit.py`  [EDITABLE — lines 214–416 only]

```python
     1: # Custom offline RL algorithm for MLS-Bench — Adroit dexterous manipulation
     2: #
     3: # EDITABLE section: network definitions + OfflineAlgorithm class.
     4: # FIXED sections: everything else (config, utilities, data, eval, training loop).
     5: import os
     6: import random
     7: import uuid
     8: from copy import deepcopy
     9: from dataclasses import dataclass
    10: from typing import Any, Dict, List, Optional, Tuple, Union
    11: 
    12: import d4rl
    13: import gym
    14: import numpy as np
    15: import pyrallis
    16: import torch
    17: import torch.nn as nn
    18: import torch.nn.functional as F
    19: from torch.distributions import Normal, TanhTransform, TransformedDistribution
    20: 
    21: TensorBatch = List[torch.Tensor]
    22: 
    23: 
    24: # =====================================================================
    25: # FIXED: Configuration
    26: # batch_size, eval_freq, n_episodes, max_timesteps are enforced here.
    27: # =====================================================================
    28: @dataclass
    29: class TrainConfig:
    30:     device: str = "cuda"
    31:     env: str = "pen-human-v1"
    32:     seed: int = 0
    33:     eval_freq: int = int(5e3)
    34:     n_episodes: int = 10
    35:     max_timesteps: int = int(1e6)
    36:     checkpoints_path: Optional[str] = None
    37:     buffer_size: int = 2_000_000
    38:     batch_size: int = 256
    39:     discount: float = 0.99
    40:     tau: float = 5e-3
    41:     actor_lr: float = 3e-4
    42:     critic_lr: float = 3e-4
    43:     alpha_lr: float = 3e-4
    44:     normalize: bool = False
    45:     orthogonal_init: bool = True
    46:     project: str = "CORL"
    47:     group: str = "custom-adroit"
    48:     name: str = "custom"
    49: 
    50:     def __post_init__(self):
    51:         self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
    52:         if self.checkpoints_path is not None:
    53:             self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)
    54: 
    55: 
    56: # =====================================================================
    57: # FIXED: Utilities
    58: # =====================================================================
    59: def soft_update(target: nn.Module, source: nn.Module, tau: float):
    60:     for tp, sp in zip(target.parameters(), source.parameters()):
    61:         tp.data.copy_((1 - tau) * tp.data + tau * sp.data)
    62: 
    63: 
    64: def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    65:     return states.mean(0), states.std(0) + eps
    66: 
    67: 
    68: def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    69:     return (states - mean) / std
    70: 
    71: 
    72: def wrap_env(
    73:     env: gym.Env,
    74:     state_mean: Union[np.ndarray, float] = 0.0,
    75:     state_std: Union[np.ndarray, float] = 1.0,
    76: ) -> gym.Env:
    77:     env = gym.wrappers.TransformObservation(env, lambda s: (s - state_mean) / state_std)
    78:     return env
    79: 
    80: 
    81: def set_seed(seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False):
    82:     if env is not None:
    83:         env.seed(seed)
    84:         env.action_space.seed(seed)
    85:     os.environ["PYTHONHASHSEED"] = str(seed)
    86:     np.random.seed(seed)
    87:     random.seed(seed)
    88:     torch.manual_seed(seed)
    89:     torch.use_deterministic_algorithms(deterministic_torch)
    90: 
    91: 
    92: def init_module_weights(module: nn.Sequential, orthogonal_init: bool = False):
    93:     if orthogonal_init:
    94:         for submodule in module[:-1]:
    95:             if isinstance(submodule, nn.Linear):
    96:                 nn.init.orthogonal_(submodule.weight, gain=np.sqrt(2))
    97:                 nn.init.constant_(submodule.bias, 0.0)
    98:     last = module[-1]
    99:     if orthogonal_init:
   100:         nn.init.orthogonal_(last.weight, gain=1e-2)
   101:     else:
   102:         nn.init.xavier_uniform_(last.weight, gain=1e-2)
   103:     nn.init.constant_(last.bias, 0.0)
   104: 
   105: 
   106: class ReplayBuffer:
   107:     def __init__(self, state_dim: int, action_dim: int, buffer_size: int, device: str = "cpu"):
   108:         self._buffer_size = buffer_size
   109:         self._pointer = 0
   110:         self._size = 0
   111:         self._states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
   112:         self._actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
   113:         self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
   114:         self._next_states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
   115:         self._next_actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
   116:         self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
   117:         self._device = device
   118: 
   119:     def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
   120:         return torch.tensor(data, dtype=torch.float32, device=self._device)
   121: 
   122:     def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
   123:         if self._size != 0:
   124:             raise ValueError("Trying to load data into non-empty replay buffer")
   125:         n = data["observations"].shape[0]
   126:         if n > self._buffer_size:
   127:             raise ValueError("Replay buffer is smaller than the dataset you are trying to load!")
   128:         self._states[:n] = self._to_tensor(data["observations"])
   129:         self._actions[:n] = self._to_tensor(data["actions"])
   130:         self._rewards[:n] = self._to_tensor(data["rewards"][..., None])
   131:         self._next_states[:n] = self._to_tensor(data["next_observations"])
   132:         self._dones[:n] = self._to_tensor(data["terminals"][..., None])
   133:         next_actions = data.get("next_actions")
   134:         if next_actions is None:
   135:             next_actions = np.concatenate([data["actions"][1:], data["actions"][-1:]], axis=0)
   136:             terminals = data["terminals"].astype(bool)
   137:             next_actions[terminals] = data["actions"][terminals]
   138:         self._next_actions[:n] = self._to_tensor(next_actions)
   139:         self._size += n
   140:         self._pointer = min(self._size, n)
   141:         print(f"Dataset size: {n}")
   142: 
   143:     def sample(self, batch_size: int) -> TensorBatch:
   144:         indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
   145:         return [
   146:             self._states[indices],
   147:             self._actions[indices],
   148:             self._rewards[indices],
   149:             self._next_states[indices],
   150:             self._dones[indices],
   151:             self._next_actions[indices],
   152:         ]
   153: 
   154: 
   155: @torch.no_grad()
   156: def eval_actor(
   157:     env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
   158: ) -> np.ndarray:
   159:     """Evaluate actor for n_episodes; returns array of episode rewards."""
   160:     env.seed(seed)
   161:     actor.eval()
   162:     episode_rewards = []
   163:     for _ in range(n_episodes):
   164:         state, done = env.reset(), False
   165:         episode_reward = 0.0
   166:         while not done:
   167:             action = actor.act(state, device)
   168:             state, reward, done, _ = env.step(action)
   169:             episode_reward += reward
   170:         episode_rewards.append(episode_reward)
   171:     actor.train()
   172:     return np.asarray(episode_rewards)
   173: 
   174: 
   175: def _mlp(input_dim: int, output_dim: int, hidden_dim: int = 256,
   176:          n_layers: int = 3) -> nn.Sequential:
   177:     """FIXED MLP factory — hidden width is locked at 256. Do NOT modify."""
   178:     assert hidden_dim == 256, "hidden_dim must be 256"
   179:     layers: list = []
   180:     layers.append(nn.Linear(input_dim, hidden_dim))
   181:     layers.append(nn.ReLU())
   182:     for _ in range(n_layers - 1):
   183:         layers.append(nn.Linear(hidden_dim, hidden_dim))
   184:         layers.append(nn.ReLU())
   185:     layers.append(nn.Linear(hidden_dim, output_dim))
   186:     return nn.Sequential(*layers)
   187: 
   188: 
   189: def _max_param_budget(state_dim: int, action_dim: int) -> int:
   190:     """Compute max parameter budget (1.3x largest baseline: EDAC/SAC-N with targets)."""
   191:     sa = state_dim + action_dim
   192:     vc = 10 * (sa * 256 + 256 + 256 * 256 + 256 + 256 * 256 + 256 + 256 + 1)
   193:     vc_target = vc
   194:     actor = (state_dim * 256 + 256 + 256 * 256 + 256 + 256 * 256 + 256
   195:              + 256 * (2 * action_dim) + 2 * action_dim)
   196:     extra = 2000
   197:     return int((vc + vc_target + actor + extra) * 1.05)
   198: 
   199: 
   200: # =====================================================================
   201: # EDITABLE: Network definitions and OfflineAlgorithm
   202: #
   203: # CONSTRAINTS:
   204: # - Total trainable parameter count is soft-capped (see budget check below).
   205: # - Total parameter count is checked at runtime and must not exceed
   206: #   1.2x the largest baseline. Focus on algorithmic improvements, not
   207: #   network capacity.
   208: #
   209: # CONFIG_OVERRIDES: set any TrainConfig field here to override the fixed
   210: # defaults. Allowed keys: normalize, normalize_reward, actor_lr, critic_lr,
   211: # tau, discount.
   212: # Example: CONFIG_OVERRIDES = {"normalize": False, "actor_lr": 1e-3}
   213: # =====================================================================
   214: CONFIG_OVERRIDES: Dict[str, Any] = {}
   215: 
   216: 
   217: class DeterministicActor(nn.Module):
   218:     """Deterministic policy pi(s) = tanh(net(s)) * max_action.
   219:     Suitable for BC, TD3+BC style algorithms. Default: 2 x 256 MLP."""
   220: 
   221:     def __init__(self, state_dim: int, action_dim: int, max_action: float):
   222:         super().__init__()
   223:         self.max_action = max_action
   224:         self.net = nn.Sequential(
   225:             nn.Linear(state_dim, 256), nn.ReLU(),
   226:             nn.Linear(256, 256), nn.ReLU(),
   227:             nn.Linear(256, action_dim), nn.Tanh(),
   228:         )
   229: 
   230:     def forward(self, state: torch.Tensor) -> torch.Tensor:
   231:         return self.max_action * self.net(state)
   232: 
   233:     @torch.no_grad()
   234:     def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
   235:         state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
   236:         return self(state).cpu().data.numpy().flatten()
   237: 
   238: 
   239: class Actor(nn.Module):
   240:     """Tanh-Gaussian stochastic policy. Default: 3 x 256 MLP.
   241:     Suitable for CQL, IQL, AWAC style algorithms."""
   242: 
   243:     def __init__(self, state_dim: int, action_dim: int, max_action: float,
   244:                  orthogonal_init: bool = False):
   245:         super().__init__()
   246:         self.max_action = max_action
   247:         self.action_dim = action_dim
   248:         self.net = nn.Sequential(
   249:             nn.Linear(state_dim, 256), nn.ReLU(),
   250:             nn.Linear(256, 256), nn.ReLU(),
   251:             nn.Linear(256, 256), nn.ReLU(),
   252:             nn.Linear(256, 2 * action_dim),
   253:         )
   254:         init_module_weights(self.net, orthogonal_init)
   255:         self.log_std_min = -20.0
   256:         self.log_std_max = 2.0
   257: 
   258:     def _get_dist(self, state: torch.Tensor):
   259:         out = self.net(state)
   260:         mean, log_std = torch.split(out, self.action_dim, dim=-1)
   261:         log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
   262:         return TransformedDistribution(
   263:             Normal(mean, torch.exp(log_std)), TanhTransform(cache_size=1)
   264:         ), mean
   265: 
   266:     def forward(self, state: torch.Tensor, deterministic: bool = False):
   267:         dist, mean = self._get_dist(state)
   268:         action = torch.tanh(mean) if deterministic else dist.rsample()
   269:         log_prob = dist.log_prob(action).sum(-1)
   270:         return self.max_action * action, log_prob
   271: 
   272:     def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
   273:         """Log-probability of a dataset action under the current policy."""
   274:         dist, _ = self._get_dist(state)
   275:         action = torch.clamp(action / self.max_action, -1.0 + 1e-6, 1.0 - 1e-6)
   276:         return dist.log_prob(action).sum(-1)
   277: 
   278:     @torch.no_grad()
   279:     def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
   280:         state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
   281:         actions, _ = self(state, not self.training)
   282:         return actions.cpu().data.numpy().flatten()
   283: 
   284: 
   285: class Critic(nn.Module):
   286:     """Q-function Q(s, a). Default: 3 x 256 MLP."""
   287: 
   288:     def __init__(self, state_dim: int, action_dim: int, orthogonal_init: bool = False):
   289:         super().__init__()
   290:         self.net = nn.Sequential(
   291:             nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
   292:             nn.Linear(256, 256), nn.ReLU(),
   293:             nn.Linear(256, 256), nn.ReLU(),
   294:             nn.Linear(256, 1),
   295:         )
   296:         init_module_weights(self.net, orthogonal_init)
   297: 
   298:     def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
   299:         return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)
   300: 
   301: 
   302: class ValueFunction(nn.Module):
   303:     """State value function V(s). Default: 3 x 256 MLP. Useful for IQL-style algorithms."""
   304: 
   305:     def __init__(self, state_dim: int, orthogonal_init: bool = False):
   306:         super().__init__()
   307:         self.net = nn.Sequential(
   308:             nn.Linear(state_dim, 256), nn.ReLU(),
   309:             nn.Linear(256, 256), nn.ReLU(),
   310:             nn.Linear(256, 256), nn.ReLU(),
   311:             nn.Linear(256, 1),
   312:         )
   313:         init_module_weights(self.net, orthogonal_init)
   314: 
   315:     def forward(self, state: torch.Tensor) -> torch.Tensor:
   316:         return self.net(state).squeeze(-1)
   317: 
   318: 
   319: class OfflineAlgorithm:
   320:     """Offline RL algorithm — implement your approach here.
   321: 
   322:     Goal: learn dexterous manipulation policies from narrow human demonstration
   323:     data on Adroit tasks (Pen, Door, Hammer with human-v1 datasets).
   324: 
   325:     Challenges specific to Adroit:
   326:     - High-dimensional action spaces (24-30 dims)
   327:     - Very few expert demonstrations (~25 human trajectories)
   328:     - Severe distribution shift between data and optimal policy
   329:     - Sparse/shaped rewards requiring precise manipulation
   330: 
   331:     The training loop calls:
   332:         trainer = OfflineAlgorithm(state_dim, action_dim, max_action, **kwargs)
   333:         log_dict = trainer.train(batch)          # called at every timestep
   334:         eval_actor(env, trainer.actor, ...)      # called every eval_freq steps
   335: 
   336:     You MUST set self.actor to an nn.Module that has an .act(state, device) method.
   337:     Build all your networks here; the training loop only provides environment dimensions
   338:     and the hyperparameters from TrainConfig.
   339: 
   340:     actor_lr, critic_lr, alpha_lr are passed from TrainConfig as starting points —
   341:     you may use different values by creating optimizers with your own lr in __init__.
   342:     Any additional algorithm-specific hyperparameters should be hardcoded in __init__.
   343: 
   344:     Dataset access:
   345:         replay_buffer is a ReplayBuffer instance containing the full offline dataset.
   346:         You can use it to compute dataset-level statistics, e.g.:
   347:             replay_buffer._states[:replay_buffer._size]   — all states  (Tensor)
   348:             replay_buffer._actions[:replay_buffer._size]  — all actions (Tensor)
   349:             replay_buffer._rewards[:replay_buffer._size]  — all rewards (Tensor)
   350:             replay_buffer._next_actions[:replay_buffer._size] — next actions (Tensor)
   351:             replay_buffer._size                           — number of transitions
   352: 
   353:     Available network classes (defined above, editable):
   354:         DeterministicActor  — deterministic tanh policy (for BC / TD3+BC approaches)
   355:         Actor               — stochastic Tanh-Gaussian policy (for CQL / IQL / AWAC approaches)
   356:         Critic              — Q(s, a) network
   357:         ValueFunction       — V(s) network (for IQL-style approaches)
   358:     Available utilities (fixed): soft_update, init_module_weights
   359:     """
   360: 
   361:     def __init__(
   362:         self,
   363:         state_dim: int,
   364:         action_dim: int,
   365:         max_action: float,
   366:         replay_buffer: "ReplayBuffer" = None,
   367:         discount: float = 0.99,
   368:         tau: float = 5e-3,
   369:         actor_lr: float = 3e-4,
   370:         critic_lr: float = 3e-4,
   371:         alpha_lr: float = 3e-4,
   372:         orthogonal_init: bool = True,
   373:         device: str = "cuda",
   374:     ):
   375:         self.device = device
   376:         self.discount = discount
   377:         self.tau = tau
   378:         self.max_action = max_action
   379:         self.total_it = 0
   380:         # Full offline dataset — use for computing global statistics
   381:         # (e.g. action mean/std for normalization, reward scaling, etc.)
   382:         self.replay_buffer = replay_buffer
   383: 
   384:         # Build networks — modify or replace as needed
   385:         self.actor = Actor(state_dim, action_dim, max_action, orthogonal_init).to(device)
   386:         self.critic_1 = Critic(state_dim, action_dim, orthogonal_init).to(device)
   387:         self.critic_2 = Critic(state_dim, action_dim, orthogonal_init).to(device)
   388:         self.critic_1_target = deepcopy(self.critic_1)
   389:         self.critic_2_target = deepcopy(self.critic_2)
   390: 
   391:         self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
   392:         self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
   393:         self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
   394: 
   395:     def train(self, batch: TensorBatch) -> Dict[str, float]:
   396:         """Update networks on one batch. Return a dict of scalar metrics for logging.
   397: 
   398:         batch = [states, actions, rewards, next_states, dones, next_actions]
   399:         (torch.Tensor, on device)
   400: 
   401:         next_actions is the action at the next timestep in the dataset
   402:         (useful for algorithms like ReBRAC that penalize deviations from
   403:         the data's next action).
   404: 
   405:         TODO: implement your offline RL algorithm here.
   406:         """
   407:         self.total_it += 1
   408:         states, actions, rewards, next_states, dones, next_actions = batch
   409: 
   410:         # ── Placeholder: replace with your algorithm ──────────────────
   411:         log_dict: Dict[str, float] = {
   412:             "actor_loss": 0.0,
   413:             "critic_loss": 0.0,
   414:         }
   415:         return log_dict
   416: 
   417: 
   418: # =====================================================================
   419: # FIXED: Training loop
   420: # =====================================================================
   421: def qlearning_dataset_with_next_actions(
   422:     env: gym.Env,
   423:     dataset: Optional[Dict[str, np.ndarray]] = None,
   424:     terminate_on_end: bool = False,
   425: ) -> Dict[str, np.ndarray]:
   426:     """CORL ReBRAC-style D4RL conversion that preserves valid next_actions."""
   427:     if dataset is None:
   428:         dataset = env.get_dataset()
   429: 
   430:     n_steps = dataset["rewards"].shape[0]
   431:     observations, next_observations = [], []
   432:     actions, next_actions = [], []
   433:     rewards, terminals = [], []
   434:     use_timeouts = "timeouts" in dataset
   435: 
   436:     episode_step = 0
   437:     for i in range(n_steps - 1):
   438:         obs = dataset["observations"][i].astype(np.float32)
   439:         next_obs = dataset["observations"][i + 1].astype(np.float32)
   440:         action = dataset["actions"][i].astype(np.float32)
   441:         next_action = dataset["actions"][i + 1].astype(np.float32)
   442:         reward = dataset["rewards"][i].astype(np.float32)
   443:         done_bool = bool(dataset["terminals"][i])
   444: 
   445:         if use_timeouts:
   446:             final_timestep = bool(dataset["timeouts"][i])
   447:         else:
   448:             final_timestep = episode_step == env._max_episode_steps - 1
   449:         if (not terminate_on_end) and final_timestep:
   450:             episode_step = 0
   451:             continue
   452:         if done_bool or final_timestep:
   453:             episode_step = 0
   454: 
   455:         observations.append(obs)
   456:         next_observations.append(next_obs)
   457:         actions.append(action)
   458:         next_actions.append(next_action)
   459:         rewards.append(reward)
   460:         terminals.append(done_bool)
   461:         episode_step += 1
   462: 
   463:     return {
   464:         "observations": np.array(observations),
   465:         "actions": np.array(actions),
   466:         "next_observations": np.array(next_observations),
   467:         "next_actions": np.array(next_actions),
   468:         "rewards": np.array(rewards),
   469:         "terminals": np.array(terminals),
   470:     }
   471: 
   472: 
   473: @pyrallis.wrap()
   474: def train(config: TrainConfig):
   475:     # Apply editable config overrides
   476:     for _k, _v in CONFIG_OVERRIDES.items():
   477:         if hasattr(config, _k):
   478:             setattr(config, _k, _v)
   479:     os.environ["ENV"] = config.env
   480: 
   481:     env = gym.make(config.env)
   482:     state_dim = env.observation_space.shape[0]
   483:     action_dim = env.action_space.shape[0]
   484:     max_action = float(env.action_space.high[0])
   485: 
   486:     dataset = qlearning_dataset_with_next_actions(env)
   487: 
   488:     if config.normalize:
   489:         state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
   490:     else:
   491:         state_mean, state_std = 0.0, 1.0
   492: 
   493:     dataset["observations"] = normalize_states(dataset["observations"], state_mean, state_std)
   494:     dataset["next_observations"] = normalize_states(
   495:         dataset["next_observations"], state_mean, state_std
   496:     )
   497:     env = wrap_env(env, state_mean=state_mean, state_std=state_std)
   498: 
   499:     replay_buffer = ReplayBuffer(state_dim, action_dim, config.buffer_size, config.device)
   500:     replay_buffer.load_d4rl_dataset(dataset)

[truncated: showing at most 500 lines / 60000 bytes from CORL/algorithms/offline/custom_adroit.py]
```




## How You Will Be Evaluated

After you finish, evaluation runs a fixed set of scripts and aggregates the
metrics they emit. These scripts are **not** in your workspace — you cannot
read or modify them. The labels below indicate what each evaluation tests:

- **pen-human-v1** — wall-clock budget `12:00:00`, compute share `0.33`
- **hammer-human-v1** — wall-clock budget `12:00:00`, compute share `0.33`
- **door-cloned-v1** — wall-clock budget `12:00:00`, compute share `0.33`


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


### `iql` baseline — editable region  [READ-ONLY — reference implementation]

In `CORL/algorithms/offline/custom_adroit.py`:

```python
Lines 214–428:
   211: # tau, discount.
   212: # Example: CONFIG_OVERRIDES = {"normalize": False, "actor_lr": 1e-3}
   213: # =====================================================================
   214: CONFIG_OVERRIDES: Dict[str, Any] = {"normalize": True}
   215: 
   216: 
   217: class DeterministicActor(nn.Module):
   218:     """Deterministic policy pi(s) = tanh(net(s)) * max_action.
   219:     Suitable for BC, TD3+BC style algorithms. Default: 2 x 256 MLP."""
   220: 
   221:     def __init__(self, state_dim: int, action_dim: int, max_action: float):
   222:         super().__init__()
   223:         self.max_action = max_action
   224:         self.net = nn.Sequential(
   225:             nn.Linear(state_dim, 256), nn.ReLU(),
   226:             nn.Linear(256, 256), nn.ReLU(),
   227:             nn.Linear(256, action_dim), nn.Tanh(),
   228:         )
   229: 
   230:     def forward(self, state: torch.Tensor) -> torch.Tensor:
   231:         return self.max_action * self.net(state)
   232: 
   233:     @torch.no_grad()
   234:     def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
   235:         state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
   236:         return self(state).cpu().data.numpy().flatten()
   237: 
   238: 
   239: class Actor(nn.Module):
   240:     """IQL GaussianPolicy — 2x256 MLP with Tanh output activation,
   241:     state-independent log_std, Normal distribution (no TanhTransform)."""
   242: 
   243:     def __init__(self, state_dim: int, action_dim: int, max_action: float,
   244:                  orthogonal_init: bool = False):
   245:         super().__init__()
   246:         self.max_action = max_action
   247:         self.action_dim = action_dim
   248:         self._mlp = nn.Sequential(
   249:             nn.Linear(state_dim, 256), nn.ReLU(), nn.Dropout(0.1),
   250:             nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.1),
   251:             nn.Linear(256, action_dim), nn.Tanh(),
   252:         )
   253:         self._log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
   254:         self._min_log_std = -20.0
   255:         self._max_log_std = 2.0
   256: 
   257:     def _get_policy(self, state: torch.Tensor):
   258:         mean = self._mlp(state)
   259:         log_std = self._log_std.clamp(self._min_log_std, self._max_log_std)
   260:         return Normal(mean, log_std.exp())
   261: 
   262:     def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
   263:         action = torch.clamp(action / self.max_action, -1.0 + 1e-6, 1.0 - 1e-6)
   264:         policy = self._get_policy(state)
   265:         return policy.log_prob(action).sum(-1)
   266: 
   267:     def forward(self, state: torch.Tensor, deterministic: bool = False):
   268:         policy = self._get_policy(state)
   269:         if deterministic:
   270:             action = policy.mean
   271:         else:
   272:             action = policy.rsample()
   273:         action = torch.clamp(action, -1.0, 1.0)
   274:         log_prob = policy.log_prob(action).sum(-1)
   275:         return self.max_action * action, log_prob
   276: 
   277:     @torch.no_grad()
   278:     def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
   279:         state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
   280:         policy = self._get_policy(state)
   281:         if self._mlp.training:
   282:             action = policy.sample()
   283:         else:
   284:             action = policy.mean
   285:         action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
   286:         return action[0].cpu().numpy()
   287: class Critic(nn.Module):
   288:     """Twin Q-function for IQL. Two 3x256 MLPs, squeeze output to scalar."""
   289: 
   290:     def __init__(self, state_dim: int, action_dim: int, orthogonal_init: bool = False):
   291:         super().__init__()
   292:         self.q1 = nn.Sequential(
   293:             nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
   294:             nn.Linear(256, 256), nn.ReLU(),
   295:             nn.Linear(256, 256), nn.ReLU(),
   296:             nn.Linear(256, 1),
   297:         )
   298:         self.q2 = nn.Sequential(
   299:             nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
   300:             nn.Linear(256, 256), nn.ReLU(),
   301:             nn.Linear(256, 256), nn.ReLU(),
   302:             nn.Linear(256, 1),
   303:         )
   304: 
   305:     def both(self, state: torch.Tensor, action: torch.Tensor):
   306:         sa = torch.cat([state, action], dim=-1)
   307:         return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)
   308: 
   309:     def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
   310:         q1, q2 = self.both(state, action)
   311:         return torch.min(q1, q2)
   312: class ValueFunction(nn.Module):
   313:     """State value function V(s). Default: 3 x 256 MLP. Useful for IQL-style algorithms."""
   314: 
   315:     def __init__(self, state_dim: int, orthogonal_init: bool = False):
   316:         super().__init__()
   317:         self.net = nn.Sequential(
   318:             nn.Linear(state_dim, 256), nn.ReLU(),
   319:             nn.Linear(256, 256), nn.ReLU(),
   320:             nn.Linear(256, 256), nn.ReLU(),
   321:             nn.Linear(256, 1),
   322:         )
   323:         init_module_weights(self.net, orthogonal_init)
   324: 
   325:     def forward(self, state: torch.Tensor) -> torch.Tensor:
   326:         return self.net(state).squeeze(-1)
   327: 
   328: 
   329: class OfflineAlgorithm:
   330:     """IQL — Implicit Q-Learning for offline RL."""
   331: 
   332:     def __init__(
   333:         self,
   334:         state_dim: int,
   335:         action_dim: int,
   336:         max_action: float,
   337:         replay_buffer=None,
   338:         discount: float = 0.99,
   339:         tau: float = 5e-3,
   340:         actor_lr: float = 3e-4,
   341:         critic_lr: float = 3e-4,
   342:         alpha_lr: float = 3e-4,
   343:         orthogonal_init: bool = True,
   344:         device: str = "cuda",
   345:     ):
   346:         self.device = device
   347:         self.discount = discount
   348:         self.tau = tau
   349:         self.max_action = max_action
   350:         self.total_it = 0
   351: 
   352:         # IQL hyperparameters (match CORL reference per-env configs)
   353:         # All adroit envs use same values; pattern supports per-env override via ENV
   354:         env_name = os.environ.get("ENV", "")
   355:         self.iql_tau = 0.8       # expectile for V loss
   356:         self.beta = 3.0          # inverse temperature for advantage weighting
   357:         self.exp_adv_max = 100.0
   358: 
   359:         # Actor (Gaussian policy with state-independent log_std)
   360:         self.actor = Actor(state_dim, action_dim, max_action).to(device)
   361:         self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
   362:         self.actor_lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
   363:             self.actor_optimizer, T_max=int(1e6)
   364:         )
   365: 
   366:         # Twin Q-network + target
   367:         self.qf = Critic(state_dim, action_dim).to(device)
   368:         self.qf_target = deepcopy(self.qf)
   369:         self.qf_target.requires_grad_(False)
   370:         self.q_optimizer = torch.optim.Adam(self.qf.parameters(), lr=critic_lr)
   371: 
   372:         # Value function V(s)
   373:         self.vf = ValueFunction(state_dim).to(device)
   374:         self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=critic_lr)
   375: 
   376:     def _asymmetric_l2_loss(self, u: torch.Tensor, tau: float) -> torch.Tensor:
   377:         return torch.mean(torch.abs(tau - (u < 0).float()) * u ** 2)
   378: 
   379:     def train(self, batch: TensorBatch) -> Dict[str, float]:
   380:         self.total_it += 1
   381:         states, actions, rewards, next_states, dones, *_ = batch
   382:         rewards = rewards.squeeze(-1)
   383:         dones = dones.squeeze(-1)
   384:         log_dict: Dict[str, float] = {}
   385: 
   386:         # ── V update: expectile regression against Q_target ──
   387:         with torch.no_grad():
   388:             target_q = self.qf_target(states, actions)
   389:         v = self.vf(states)
   390:         adv = target_q - v
   391:         v_loss = self._asymmetric_l2_loss(adv, self.iql_tau)
   392:         log_dict["value_loss"] = v_loss.item()
   393: 
   394:         self.v_optimizer.zero_grad()
   395:         v_loss.backward()
   396:         self.v_optimizer.step()
   397: 
   398:         # ── Q update: Bellman with V(s') as bootstrap ──
   399:         with torch.no_grad():
   400:             next_v = self.vf(next_states)
   401:             q_target = rewards + (1.0 - dones) * self.discount * next_v
   402: 
   403:         q1, q2 = self.qf.both(states, actions)
   404:         q_loss = (F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)) / 2.0
   405:         log_dict["critic_loss"] = q_loss.item()
   406: 
   407:         self.q_optimizer.zero_grad()
   408:         q_loss.backward()
   409:         self.q_optimizer.step()
   410: 
   411:         # Target Q update
   412:         soft_update(self.qf_target, self.qf, self.tau)
   413: 
   414:         # ── Actor update: advantage-weighted regression ──
   415:         with torch.no_grad():
   416:             adv_detached = target_q - self.vf(states)
   417:             exp_adv = torch.exp(self.beta * adv_detached).clamp(max=self.exp_adv_max)
   418: 
   419:         action_log_prob = self.actor.log_prob(states, actions)
   420:         actor_loss = torch.mean(exp_adv * (-action_log_prob))
   421:         log_dict["actor_loss"] = actor_loss.item()
   422: 
   423:         self.actor_optimizer.zero_grad()
   424:         actor_loss.backward()
   425:         self.actor_optimizer.step()
   426:         self.actor_lr_schedule.step()
   427: 
   428:         return log_dict
   429: 
   430: # =====================================================================
   431: # FIXED: Training loop
```

### `awac` baseline — editable region  [READ-ONLY — reference implementation]

In `CORL/algorithms/offline/custom_adroit.py`:

```python
Lines 214–410:
   211: # tau, discount.
   212: # Example: CONFIG_OVERRIDES = {"normalize": False, "actor_lr": 1e-3}
   213: # =====================================================================
   214: CONFIG_OVERRIDES: Dict[str, Any] = {"normalize": True}
   215: 
   216: 
   217: class DeterministicActor(nn.Module):
   218:     """Deterministic policy pi(s) = tanh(net(s)) * max_action.
   219:     Suitable for BC, TD3+BC style algorithms. Default: 2 x 256 MLP."""
   220: 
   221:     def __init__(self, state_dim: int, action_dim: int, max_action: float):
   222:         super().__init__()
   223:         self.max_action = max_action
   224:         self.net = nn.Sequential(
   225:             nn.Linear(state_dim, 256), nn.ReLU(),
   226:             nn.Linear(256, 256), nn.ReLU(),
   227:             nn.Linear(256, action_dim), nn.Tanh(),
   228:         )
   229: 
   230:     def forward(self, state: torch.Tensor) -> torch.Tensor:
   231:         return self.max_action * self.net(state)
   232: 
   233:     @torch.no_grad()
   234:     def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
   235:         state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
   236:         return self(state).cpu().data.numpy().flatten()
   237: 
   238: 
   239: class Actor(nn.Module):
   240:     """AWAC GaussianPolicy — 3x256 MLP, state-independent log_std, Normal + clamp."""
   241: 
   242:     def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
   243:                  min_log_std: float = -20.0, max_log_std: float = 2.0):
   244:         super().__init__()
   245:         self._mlp = nn.Sequential(
   246:             nn.Linear(state_dim, hidden_dim),
   247:             nn.ReLU(),
   248:             nn.Linear(hidden_dim, hidden_dim),
   249:             nn.ReLU(),
   250:             nn.Linear(hidden_dim, hidden_dim),
   251:             nn.ReLU(),
   252:             nn.Linear(hidden_dim, action_dim),
   253:         )
   254:         self._log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
   255:         self._min_log_std = min_log_std
   256:         self._max_log_std = max_log_std
   257: 
   258:     def _get_policy(self, state: torch.Tensor):
   259:         mean = self._mlp(state)
   260:         log_std = self._log_std.clamp(self._min_log_std, self._max_log_std)
   261:         return torch.distributions.Normal(mean, log_std.exp())
   262: 
   263:     def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
   264:         policy = self._get_policy(state)
   265:         return policy.log_prob(action).sum(-1, keepdim=True)
   266: 
   267:     def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
   268:         policy = self._get_policy(state)
   269:         action = policy.rsample()
   270:         action.clamp_(-1.0, 1.0)
   271:         log_prob = policy.log_prob(action).sum(-1, keepdim=True)
   272:         return action, log_prob
   273: 
   274:     def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
   275:         state_t = torch.tensor(state[None], dtype=torch.float32, device=device)
   276:         policy = self._get_policy(state_t)
   277:         if self._mlp.training:
   278:             action_t = policy.sample()
   279:         else:
   280:             action_t = policy.mean
   281:         return action_t[0].cpu().numpy()
   282: class Critic(nn.Module):
   283:     """Q-function Q(s, a). 3x256 MLP, returns (batch, 1)."""
   284: 
   285:     def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
   286:         super().__init__()
   287:         self._mlp = nn.Sequential(
   288:             nn.Linear(state_dim + action_dim, hidden_dim),
   289:             nn.ReLU(),
   290:             nn.Linear(hidden_dim, hidden_dim),
   291:             nn.ReLU(),
   292:             nn.Linear(hidden_dim, hidden_dim),
   293:             nn.ReLU(),
   294:             nn.Linear(hidden_dim, 1),
   295:         )
   296: 
   297:     def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
   298:         return self._mlp(torch.cat([state, action], dim=-1))
   299: class ValueFunction(nn.Module):
   300:     """State value function V(s). Default: 3 x 256 MLP. Useful for IQL-style algorithms."""
   301: 
   302:     def __init__(self, state_dim: int, orthogonal_init: bool = False):
   303:         super().__init__()
   304:         self.net = nn.Sequential(
   305:             nn.Linear(state_dim, 256), nn.ReLU(),
   306:             nn.Linear(256, 256), nn.ReLU(),
   307:             nn.Linear(256, 256), nn.ReLU(),
   308:             nn.Linear(256, 1),
   309:         )
   310:         init_module_weights(self.net, orthogonal_init)
   311: 
   312:     def forward(self, state: torch.Tensor) -> torch.Tensor:
   313:         return self.net(state).squeeze(-1)
   314: 
   315: 
   316: class OfflineAlgorithm:
   317:     """AWAC — Advantage Weighted Actor-Critic for offline RL."""
   318: 
   319:     def __init__(
   320:         self,
   321:         state_dim: int,
   322:         action_dim: int,
   323:         max_action: float,
   324:         replay_buffer=None,
   325:         discount: float = 0.99,
   326:         tau: float = 5e-3,
   327:         actor_lr: float = 3e-4,
   328:         critic_lr: float = 3e-4,
   329:         alpha_lr: float = 3e-4,
   330:         orthogonal_init: bool = True,
   331:         device: str = "cuda",
   332:     ):
   333:         self.device = device
   334:         self.discount = discount
   335:         self.tau = tau
   336:         self.max_action = max_action
   337:         self.total_it = 0
   338: 
   339:         # AWAC hyperparameters (match CORL reference: awac_lambda=0.1)
   340:         self.awac_lambda = 0.1
   341:         self.exp_adv_max = 100.0
   342: 
   343:         # Actor (GaussianPolicy-style with state-independent log_std)
   344:         self.actor = Actor(state_dim, action_dim, 256).to(device)
   345:         self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
   346: 
   347:         # Twin critics + targets (SEPARATE optimizers)
   348:         self.critic_1 = Critic(state_dim, action_dim, 256).to(device)
   349:         self.critic_2 = Critic(state_dim, action_dim, 256).to(device)
   350:         self.target_critic_1 = deepcopy(self.critic_1)
   351:         self.target_critic_2 = deepcopy(self.critic_2)
   352:         self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
   353:         self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
   354: 
   355:     def train(self, batch: TensorBatch) -> Dict[str, float]:
   356:         self.total_it += 1
   357:         states, actions, rewards, next_states, dones, *_ = batch
   358:         log_dict: Dict[str, float] = {}
   359: 
   360:         # Critic update
   361:         with torch.no_grad():
   362:             next_actions, _ = self.actor(next_states)
   363:             q_next = torch.min(
   364:                 self.target_critic_1(next_states, next_actions),
   365:                 self.target_critic_2(next_states, next_actions),
   366:             )
   367:             q_target = rewards + self.discount * (1.0 - dones) * q_next
   368: 
   369:         q1 = self.critic_1(states, actions)
   370:         q2 = self.critic_2(states, actions)
   371:         q1_loss = F.mse_loss(q1, q_target)
   372:         q2_loss = F.mse_loss(q2, q_target)
   373:         critic_loss = q1_loss + q2_loss
   374:         log_dict["critic_loss"] = critic_loss.item()
   375: 
   376:         self.critic_1_optimizer.zero_grad()
   377:         self.critic_2_optimizer.zero_grad()
   378:         critic_loss.backward()
   379:         self.critic_1_optimizer.step()
   380:         self.critic_2_optimizer.step()
   381: 
   382:         # Actor update (advantage-weighted)
   383:         with torch.no_grad():
   384:             pi_action, _ = self.actor(states)
   385:             v = torch.min(
   386:                 self.critic_1(states, pi_action),
   387:                 self.critic_2(states, pi_action),
   388:             )
   389:             q = torch.min(
   390:                 self.critic_1(states, actions),
   391:                 self.critic_2(states, actions),
   392:             )
   393:             adv = q - v
   394:             weights = torch.clamp_max(
   395:                 torch.exp(adv / self.awac_lambda), self.exp_adv_max
   396:             )
   397: 
   398:         action_log_prob = self.actor.log_prob(states, actions)
   399:         actor_loss = (-action_log_prob * weights).mean()
   400:         log_dict["actor_loss"] = actor_loss.item()
   401: 
   402:         self.actor_optimizer.zero_grad()
   403:         actor_loss.backward()
   404:         self.actor_optimizer.step()
   405: 
   406:         # Target update
   407:         soft_update(self.target_critic_1, self.critic_1, self.tau)
   408:         soft_update(self.target_critic_2, self.critic_2, self.tau)
   409: 
   410:         return log_dict
   411: 
   412: # =====================================================================
   413: # FIXED: Training loop
```

### `rebrac` baseline — editable region  [READ-ONLY — reference implementation]

In `CORL/algorithms/offline/custom_adroit.py`:

```python
Lines 214–453:
   211: # tau, discount.
   212: # Example: CONFIG_OVERRIDES = {"normalize": False, "actor_lr": 1e-3}
   213: # =====================================================================
   214: CONFIG_OVERRIDES: Dict[str, Any] = {}
   215: 
   216: 
   217: class DeterministicActor(nn.Module):
   218:     """Deterministic policy pi(s) = tanh(net(s)) * max_action.
   219:     Suitable for BC, TD3+BC style algorithms. Default: 2 x 256 MLP."""
   220: 
   221:     def __init__(self, state_dim: int, action_dim: int, max_action: float):
   222:         super().__init__()
   223:         self.max_action = max_action
   224:         self.net = nn.Sequential(
   225:             nn.Linear(state_dim, 256), nn.ReLU(),
   226:             nn.Linear(256, 256), nn.ReLU(),
   227:             nn.Linear(256, 256), nn.ReLU(),
   228:             nn.Linear(256, action_dim), nn.Tanh(),
   229:         )
   230:         # CORL-style init: pytorch_init for hidden, uniform_init(1e-3) for output
   231:         import math
   232:         for i, layer in enumerate(self.net):
   233:             if isinstance(layer, nn.Linear):
   234:                 fan_in = layer.in_features
   235:                 if i < len(self.net) - 2:  # hidden layers
   236:                     bound = math.sqrt(1.0 / fan_in)
   237:                     nn.init.uniform_(layer.weight, -bound, bound)
   238:                     nn.init.constant_(layer.bias, 0.1)
   239:                 else:  # output layer
   240:                     nn.init.uniform_(layer.weight, -1e-3, 1e-3)
   241:                     nn.init.uniform_(layer.bias, -1e-3, 1e-3)
   242: 
   243:     def forward(self, state: torch.Tensor) -> torch.Tensor:
   244:         return self.max_action * self.net(state)
   245: 
   246:     @torch.no_grad()
   247:     def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
   248:         state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
   249:         return self(state).cpu().data.numpy().flatten()
   250: 
   251: class Actor(nn.Module):
   252:     """Tanh-Gaussian stochastic policy. Default: 3 x 256 MLP.
   253:     Suitable for CQL, IQL, AWAC style algorithms."""
   254: 
   255:     def __init__(self, state_dim: int, action_dim: int, max_action: float,
   256:                  orthogonal_init: bool = False):
   257:         super().__init__()
   258:         self.max_action = max_action
   259:         self.action_dim = action_dim
   260:         self.net = nn.Sequential(
   261:             nn.Linear(state_dim, 256), nn.ReLU(),
   262:             nn.Linear(256, 256), nn.ReLU(),
   263:             nn.Linear(256, 256), nn.ReLU(),
   264:             nn.Linear(256, 2 * action_dim),
   265:         )
   266:         init_module_weights(self.net, orthogonal_init)
   267:         self.log_std_min = -20.0
   268:         self.log_std_max = 2.0
   269: 
   270:     def _get_dist(self, state: torch.Tensor):
   271:         out = self.net(state)
   272:         mean, log_std = torch.split(out, self.action_dim, dim=-1)
   273:         log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
   274:         return TransformedDistribution(
   275:             Normal(mean, torch.exp(log_std)), TanhTransform(cache_size=1)
   276:         ), mean
   277: 
   278:     def forward(self, state: torch.Tensor, deterministic: bool = False):
   279:         dist, mean = self._get_dist(state)
   280:         action = torch.tanh(mean) if deterministic else dist.rsample()
   281:         log_prob = dist.log_prob(action).sum(-1)
   282:         return self.max_action * action, log_prob
   283: 
   284:     def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
   285:         """Log-probability of a dataset action under the current policy."""
   286:         dist, _ = self._get_dist(state)
   287:         action = torch.clamp(action / self.max_action, -1.0 + 1e-6, 1.0 - 1e-6)
   288:         return dist.log_prob(action).sum(-1)
   289: 
   290:     @torch.no_grad()
   291:     def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
   292:         state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
   293:         actions, _ = self(state, not self.training)
   294:         return actions.cpu().data.numpy().flatten()
   295: 
   296: 
   297: class Critic(nn.Module):
   298:     """Q-function with post-activation LayerNorm (ReBRAC critic_ln=True). 3x256 MLP."""
   299: 
   300:     def __init__(self, state_dim: int, action_dim: int, orthogonal_init: bool = False):
   301:         super().__init__()
   302:         self.net = nn.Sequential(
   303:             nn.Linear(state_dim + action_dim, 256), nn.ReLU(), nn.LayerNorm(256),
   304:             nn.Linear(256, 256), nn.ReLU(), nn.LayerNorm(256),
   305:             nn.Linear(256, 256), nn.ReLU(), nn.LayerNorm(256),
   306:             nn.Linear(256, 1),
   307:         )
   308:         # CORL-style init: pytorch_init for hidden, uniform_init(3e-3) for output
   309:         import math
   310:         for i, layer in enumerate(self.net):
   311:             if isinstance(layer, nn.Linear):
   312:                 fan_in = layer.in_features
   313:                 if i < len(self.net) - 1:  # hidden layers
   314:                     bound = math.sqrt(1.0 / fan_in)
   315:                     nn.init.uniform_(layer.weight, -bound, bound)
   316:                     nn.init.constant_(layer.bias, 0.1)
   317:                 else:  # output layer
   318:                     nn.init.uniform_(layer.weight, -3e-3, 3e-3)
   319:                     nn.init.uniform_(layer.bias, -3e-3, 3e-3)
   320: 
   321:     def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
   322:         return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)
   323: class ValueFunction(nn.Module):
   324:     """State value function V(s). Default: 3 x 256 MLP. Useful for IQL-style algorithms."""
   325: 
   326:     def __init__(self, state_dim: int, orthogonal_init: bool = False):
   327:         super().__init__()
   328:         self.net = nn.Sequential(
   329:             nn.Linear(state_dim, 256), nn.ReLU(),
   330:             nn.Linear(256, 256), nn.ReLU(),
   331:             nn.Linear(256, 256), nn.ReLU(),
   332:             nn.Linear(256, 1),
   333:         )
   334:         init_module_weights(self.net, orthogonal_init)
   335: 
   336:     def forward(self, state: torch.Tensor) -> torch.Tensor:
   337:         return self.net(state).squeeze(-1)
   338: 
   339: 
   340: class OfflineAlgorithm:
   341:     """ReBRAC — TD3+BC with critic BC regularization in Bellman target."""
   342: 
   343:     def __init__(
   344:         self,
   345:         state_dim: int,
   346:         action_dim: int,
   347:         max_action: float,
   348:         replay_buffer=None,
   349:         discount: float = 0.99,
   350:         tau: float = 5e-3,
   351:         actor_lr: float = 3e-4,
   352:         critic_lr: float = 3e-4,
   353:         alpha_lr: float = 3e-4,
   354:         orthogonal_init: bool = True,
   355:         device: str = "cuda",
   356:     ):
   357:         self.device = device
   358:         self.discount = discount
   359:         self.tau = tau
   360:         self.max_action = max_action
   361:         self.total_it = 0
   362: 
   363:         # ReBRAC hyperparameters (per-env from CORL reference configs)
   364:         env_name = os.environ.get("ENV", "")
   365:         if "hammer" in env_name:
   366:             self.actor_bc_coef = 0.01
   367:             self.critic_bc_coef = 0.5
   368:         elif "door-cloned" in env_name:
   369:             self.actor_bc_coef = 0.01
   370:             self.critic_bc_coef = 0.1
   371:         elif "door" in env_name:
   372:             self.actor_bc_coef = 0.1
   373:             self.critic_bc_coef = 0.1
   374:         else:  # pen (default)
   375:             self.actor_bc_coef = 0.1
   376:             self.critic_bc_coef = 0.5
   377:         self.policy_noise = 0.2
   378:         self.noise_clip = 0.5
   379:         self.policy_freq = 2
   380:         self.normalize_q = True
   381: 
   382:         # Actor (deterministic, 3x256, NO LayerNorm)
   383:         self.actor = DeterministicActor(state_dim, action_dim, max_action).to(device)
   384:         self.actor_target = deepcopy(self.actor)
   385:         self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
   386: 
   387:         # Twin critics (3x256, WITH post-activation LayerNorm) + targets
   388:         self.critic_1 = Critic(state_dim, action_dim).to(device)
   389:         self.critic_1_target = deepcopy(self.critic_1)
   390:         self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=3e-4)
   391: 
   392:         self.critic_2 = Critic(state_dim, action_dim).to(device)
   393:         self.critic_2_target = deepcopy(self.critic_2)
   394:         self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=3e-4)
   395: 
   396:     def train(self, batch: TensorBatch) -> Dict[str, float]:
   397:         self.total_it += 1
   398:         states, actions, rewards, next_states, dones, next_actions_data = batch
   399:         rewards = rewards.squeeze(-1)
   400:         dones = dones.squeeze(-1)
   401:         log_dict: Dict[str, float] = {}
   402: 
   403:         # Critic update
   404:         with torch.no_grad():
   405:             noise = (torch.randn_like(actions) * self.policy_noise).clamp(
   406:                 -self.noise_clip, self.noise_clip
   407:             )
   408:             next_actions_policy = (self.actor_target(next_states) + noise).clamp(-1.0, 1.0)
   409: 
   410:             # Critic BC: subtract penalty from next_q
   411:             bc_penalty = ((next_actions_policy - next_actions_data) ** 2).sum(-1)
   412:             target_q1 = self.critic_1_target(next_states, next_actions_policy)
   413:             target_q2 = self.critic_2_target(next_states, next_actions_policy)
   414:             next_q = torch.min(target_q1, target_q2)
   415:             next_q = next_q - self.critic_bc_coef * bc_penalty
   416:             target_q = rewards + (1.0 - dones) * self.discount * next_q
   417: 
   418:         q1 = self.critic_1(states, actions)
   419:         q2 = self.critic_2(states, actions)
   420:         critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
   421:         log_dict["critic_loss"] = critic_loss.item()
   422: 
   423:         self.critic_1_optimizer.zero_grad()
   424:         self.critic_2_optimizer.zero_grad()
   425:         critic_loss.backward()
   426:         self.critic_1_optimizer.step()
   427:         self.critic_2_optimizer.step()
   428: 
   429:         # Delayed actor update
   430:         if self.total_it % self.policy_freq == 0:
   431:             pi = self.actor(states)
   432:             bc_penalty_actor = ((pi - actions) ** 2).sum(-1)
   433:             q_values = torch.min(
   434:                 self.critic_1(states, pi),
   435:                 self.critic_2(states, pi),
   436:             )
   437: 
   438:             lmbda = 1.0
   439:             if self.normalize_q:
   440:                 lmbda = 1.0 / q_values.abs().mean().detach()
   441: 
   442:             actor_loss = (self.actor_bc_coef * bc_penalty_actor - lmbda * q_values).mean()
   443:             log_dict["actor_loss"] = actor_loss.item()
   444: 
   445:             self.actor_optimizer.zero_grad()
   446:             actor_loss.backward()
   447:             self.actor_optimizer.step()
   448: 
   449:             soft_update(self.critic_1_target, self.critic_1, self.tau)
   450:             soft_update(self.critic_2_target, self.critic_2, self.tau)
   451:             soft_update(self.actor_target, self.actor, self.tau)
   452: 
   453:         return log_dict
   454: 
   455: # =====================================================================
   456: # FIXED: Training loop
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
