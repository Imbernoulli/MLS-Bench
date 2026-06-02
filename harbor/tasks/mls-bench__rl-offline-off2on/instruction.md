# MLS-Bench: rl-offline-off2on

# Offline-to-Online RL: Preventing Catastrophic Forgetting in Fine-Tuning

## Research Question
Design and implement an offline-to-online RL algorithm that pretrains
from an offline dataset and then fine-tunes with online interaction
without catastrophic forgetting or Q-value collapse. Your code goes in
`custom_finetune.py`. Several reference implementations are provided as
read-only `*.edit.py` baselines.

## Background
Offline-to-online RL pretrains a policy and value function on a fixed
dataset and then continues learning with environment interaction. The
offline-to-online transition is brittle: conservative offline value
functions can become overoptimistic once online data shifts the replay
distribution, behavior-regularized policies can forget useful offline
behavior, and naive fine-tuning often causes a Q-value collapse and a
performance drop early in the online phase.

The Adroit `cloned-v1` datasets mix expert and noisy demonstrations, so
the offline pretraining never produces a strong policy on its own and
the online phase must improve substantially without losing what little
competence was learned.

Reference baselines spanning the design space:
- **AWAC** — Nair et al., "AWAC: Accelerating Online Reinforcement
  Learning with Offline Datasets" (arXiv:2006.09359). Implicit
  advantage-weighted policy constraint that allows smooth fine-tuning.
  Default Lagrange temperature `lambda = 1.0`.
- **SPOT** — Wu et al., "Supported Policy Optimization for Offline
  Reinforcement Learning" (arXiv:2202.06239, NeurIPS 2022). VAE-based
  density support constraint that supports online fine-tuning.
- **IQL** — Kostrikov et al., "Offline Reinforcement Learning with
  Implicit Q-Learning" (arXiv:2110.06169, ICLR 2022). Expectile
  regression pretraining with advantage-weighted policy extraction,
  providing a stable offline initialization for online fine-tuning.

## Constraints
- **Network dimensions are fixed at 256.** All MLP hidden layers must
  use 256 units. A `_mlp()` factory function is provided in the FIXED
  section for convenience. You may define custom network classes but
  hidden widths must remain 256.
- **Total parameter count is enforced.** The training loop checks that
  total trainable parameters do not exceed 1.2x the largest baseline
  architecture, so the contribution must be algorithmic (transition
  handling, value calibration, replay balancing, behavior-constraint
  annealing) rather than capacity.
- Do **not** simply copy a reference implementation with minor changes.

## Evaluation
Trained and evaluated on Adroit tasks Pen, Door and Hammer using the
D4RL `cloned-v1` datasets. The offline phase trains for 1M gradient
steps, then the online phase trains for 1M environment-interaction
steps. Metric: D4RL normalized score (0 = random, 100 = expert),
evaluated throughout both phases. Higher is better. Strong methods
should retain offline competence while benefiting from online
fine-tuning across the manipulation tasks.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/CORL/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `CORL/algorithms/finetune/custom_finetune.py`
- editable lines **258–477**




## Readable Context


### `CORL/algorithms/finetune/custom_finetune.py`  [EDITABLE — lines 258–477 only]

```python
     1: # Custom offline-to-online RL algorithm for MLS-Bench — Adroit fine-tuning
     2: #
     3: # EDITABLE section: network definitions + OfflineOnlineAlgorithm class.
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
    23: ENVS_WITH_GOAL = ("pen", "door", "hammer", "relocate", "antmaze")
    24: 
    25: 
    26: # =====================================================================
    27: # FIXED: Configuration
    28: # =====================================================================
    29: @dataclass
    30: class TrainConfig:
    31:     device: str = "cuda"
    32:     env: str = "pen-cloned-v1"
    33:     seed: int = 0
    34:     eval_seed: int = 0
    35:     eval_freq: int = int(5e3)
    36:     n_episodes: int = 10
    37:     offline_iterations: int = int(1e6)
    38:     online_iterations: int = int(1e6)
    39:     checkpoints_path: Optional[str] = None
    40:     buffer_size: int = 20_000_000
    41:     batch_size: int = 256
    42:     discount: float = 0.99
    43:     tau: float = 5e-3
    44:     actor_lr: float = 3e-4
    45:     critic_lr: float = 3e-4
    46:     expl_noise: float = 0.1
    47:     normalize: bool = True
    48:     normalize_reward: bool = False
    49:     project: str = "CORL"
    50:     group: str = "custom-finetune"
    51:     name: str = "custom"
    52: 
    53:     def __post_init__(self):
    54:         self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
    55:         if self.checkpoints_path is not None:
    56:             self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)
    57: 
    58: 
    59: # =====================================================================
    60: # FIXED: Utilities
    61: # =====================================================================
    62: def soft_update(target: nn.Module, source: nn.Module, tau: float):
    63:     for tp, sp in zip(target.parameters(), source.parameters()):
    64:         tp.data.copy_((1 - tau) * tp.data + tau * sp.data)
    65: 
    66: 
    67: def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    68:     return states.mean(0), states.std(0) + eps
    69: 
    70: 
    71: def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    72:     return (states - mean) / std
    73: 
    74: 
    75: def wrap_env(
    76:     env: gym.Env,
    77:     state_mean: Union[np.ndarray, float] = 0.0,
    78:     state_std: Union[np.ndarray, float] = 1.0,
    79: ) -> gym.Env:
    80:     env = gym.wrappers.TransformObservation(env, lambda s: (s - state_mean) / state_std)
    81:     return env
    82: 
    83: 
    84: def set_seed(seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False):
    85:     if env is not None:
    86:         env.seed(seed)
    87:         env.action_space.seed(seed)
    88:     os.environ["PYTHONHASHSEED"] = str(seed)
    89:     np.random.seed(seed)
    90:     random.seed(seed)
    91:     torch.manual_seed(seed)
    92:     torch.use_deterministic_algorithms(deterministic_torch)
    93: 
    94: 
    95: def set_env_seed(env: gym.Env, seed: int):
    96:     env.seed(seed)
    97:     env.action_space.seed(seed)
    98: 
    99: 
   100: def is_goal_reached(reward: float, info: dict) -> bool:
   101:     if "goal_achieved" in info:
   102:         return info["goal_achieved"]
   103:     return reward > 0
   104: 
   105: 
   106: def init_module_weights(module: nn.Sequential, orthogonal_init: bool = False):
   107:     if orthogonal_init:
   108:         for submodule in module[:-1]:
   109:             if isinstance(submodule, nn.Linear):
   110:                 nn.init.orthogonal_(submodule.weight, gain=np.sqrt(2))
   111:                 nn.init.constant_(submodule.bias, 0.0)
   112:     last = module[-1]
   113:     if orthogonal_init:
   114:         nn.init.orthogonal_(last.weight, gain=1e-2)
   115:     else:
   116:         nn.init.xavier_uniform_(last.weight, gain=1e-2)
   117:     nn.init.constant_(last.bias, 0.0)
   118: 
   119: 
   120: class ReplayBuffer:
   121:     def __init__(self, state_dim: int, action_dim: int, buffer_size: int, device: str = "cpu"):
   122:         self._buffer_size = buffer_size
   123:         self._pointer = 0
   124:         self._size = 0
   125:         self._states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
   126:         self._actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
   127:         self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
   128:         self._next_states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
   129:         self._next_actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
   130:         self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
   131:         self._device = device
   132: 
   133:     def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
   134:         return torch.tensor(data, dtype=torch.float32, device=self._device)
   135: 
   136:     def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
   137:         if self._size != 0:
   138:             raise ValueError("Trying to load data into non-empty replay buffer")
   139:         n = data["observations"].shape[0]
   140:         if n > self._buffer_size:
   141:             raise ValueError("Replay buffer is smaller than the dataset you are trying to load!")
   142:         self._states[:n] = self._to_tensor(data["observations"])
   143:         self._actions[:n] = self._to_tensor(data["actions"])
   144:         self._rewards[:n] = self._to_tensor(data["rewards"][..., None])
   145:         self._next_states[:n] = self._to_tensor(data["next_observations"])
   146:         self._dones[:n] = self._to_tensor(data["terminals"][..., None])
   147:         # Compute next_actions: action at the next timestep in the dataset.
   148:         # At episode boundaries (terminal=1) or the last transition, use the current action.
   149:         next_actions = np.concatenate([data["actions"][1:], data["actions"][-1:]], axis=0)
   150:         terminals = data["terminals"].astype(bool)
   151:         next_actions[terminals] = data["actions"][terminals]
   152:         self._next_actions[:n] = self._to_tensor(next_actions)
   153:         self._size += n
   154:         self._pointer = min(self._size, n)
   155:         print(f"Dataset size: {n}")
   156: 
   157:     def add_transition(
   158:         self, state: np.ndarray, action: np.ndarray, reward: float,
   159:         next_state: np.ndarray, done: bool,
   160:     ):
   161:         self._states[self._pointer] = self._to_tensor(state)
   162:         self._actions[self._pointer] = self._to_tensor(action)
   163:         self._rewards[self._pointer] = self._to_tensor(np.array([reward]))
   164:         self._next_states[self._pointer] = self._to_tensor(next_state)
   165:         self._next_actions[self._pointer] = self._to_tensor(action)  # placeholder for online transitions
   166:         self._dones[self._pointer] = self._to_tensor(np.array([float(done)]))
   167:         self._pointer = (self._pointer + 1) % self._buffer_size
   168:         self._size = min(self._size + 1, self._buffer_size)
   169: 
   170:     def sample(self, batch_size: int) -> TensorBatch:
   171:         indices = np.random.randint(0, min(self._size, self._buffer_size), size=batch_size)
   172:         return [
   173:             self._states[indices],
   174:             self._actions[indices],
   175:             self._rewards[indices],
   176:             self._next_states[indices],
   177:             self._dones[indices],
   178:             self._next_actions[indices],
   179:         ]
   180: 
   181: 
   182: @torch.no_grad()
   183: def eval_actor(
   184:     env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
   185: ) -> Tuple[np.ndarray, float]:
   186:     """Evaluate actor for n_episodes; returns (episode_rewards, success_rate)."""
   187:     env.seed(seed)
   188:     actor.eval()
   189:     episode_rewards = []
   190:     successes = []
   191:     for _ in range(n_episodes):
   192:         state, done = env.reset(), False
   193:         episode_reward = 0.0
   194:         goal_achieved = False
   195:         while not done:
   196:             action = actor.act(state, device)
   197:             state, reward, done, info = env.step(action)
   198:             episode_reward += reward
   199:             if not goal_achieved:
   200:                 goal_achieved = is_goal_reached(reward, info)
   201:         episode_rewards.append(episode_reward)
   202:         successes.append(float(goal_achieved))
   203:     actor.train()
   204:     return np.asarray(episode_rewards), np.mean(successes)
   205: 
   206: 
   207: def _mlp(input_dim: int, output_dim: int, hidden_dim: int = 256,
   208:          n_layers: int = 3) -> nn.Sequential:
   209:     """FIXED MLP factory — hidden width is locked at 256. Do NOT modify."""
   210:     assert hidden_dim == 256, "hidden_dim must be 256"
   211:     layers: list = []
   212:     layers.append(nn.Linear(input_dim, hidden_dim))
   213:     layers.append(nn.ReLU())
   214:     for _ in range(n_layers - 1):
   215:         layers.append(nn.Linear(hidden_dim, hidden_dim))
   216:         layers.append(nn.ReLU())
   217:     layers.append(nn.Linear(hidden_dim, output_dim))
   218:     return nn.Sequential(*layers)
   219: 
   220: 
   221: def _max_param_budget(state_dim: int, action_dim: int) -> int:
   222:     """Compute max parameter budget (1.2x largest baseline: CQL/Cal-QL + VAE)."""
   223:     sa = state_dim + action_dim
   224:     # Two critics: 3 hidden layers of 256, output 1
   225:     critics = 2 * (sa * 256 + 256 + 256 * 256 + 256 + 256 * 256 + 256 + 256 + 1)
   226:     # Stochastic actor: 3 hidden layers, output 2*action_dim + Scalar params
   227:     actor = (state_dim * 256 + 256 + 256 * 256 + 256 + 256 * 256 + 256
   228:              + 256 * (2 * action_dim) + 2 * action_dim + 10)
   229:     # Value function: 3 hidden layers, output 1
   230:     vf = state_dim * 256 + 256 + 256 * 256 + 256 + 256 * 256 + 256 + 256 + 1
   231:     # VAE (SPOT baseline uses hidden_dim=750): encoder + decoder
   232:     vae_latent = 2 * action_dim
   233:     vae_enc = (sa * 750 + 750 + 750 * 750 + 750 + 750 * vae_latent + vae_latent
   234:                + 750 * vae_latent + vae_latent)
   235:     vae_dec = ((state_dim + vae_latent) * 750 + 750 + 750 * 750 + 750
   236:                + 750 * action_dim + action_dim)
   237:     vae = vae_enc + vae_dec
   238:     extra = 5000  # Scalar params, log_alpha, mc_returns, etc.
   239:     # Include target critics (separate copy)
   240:     critic_targets = critics
   241:     return int((critics + critic_targets + actor + vf + vae + extra) * 1.05)
   242: 
   243: 
   244: # =====================================================================
   245: # EDITABLE: Network definitions and OfflineOnlineAlgorithm
   246: #
   247: # CONSTRAINTS:
   248: # - Total trainable parameter count is soft-capped (see budget check below).
   249: # - Total parameter count is checked at runtime and must not exceed
   250: #   1.2x the largest baseline. Focus on algorithmic improvements, not
   251: #   network capacity.
   252: #
   253: # CONFIG_OVERRIDES: override method-specific TrainConfig fields here.
   254: # Allowed keys: normalize, normalize_reward, actor_lr, critic_lr, tau,
   255: # expl_noise, discount.
   256: # Example: CONFIG_OVERRIDES = {"normalize": False, "actor_lr": 1e-3}
   257: # =====================================================================
   258: CONFIG_OVERRIDES: Dict[str, Any] = {}
   259: class DeterministicActor(nn.Module):
   260:     """Deterministic policy pi(s) = tanh(net(s)) * max_action.
   261:     Suitable for TD3+BC / SPOT style algorithms. Default: 2 x 256 MLP."""
   262: 
   263:     def __init__(self, state_dim: int, action_dim: int, max_action: float):
   264:         super().__init__()
   265:         self.max_action = max_action
   266:         self.net = nn.Sequential(
   267:             nn.Linear(state_dim, 256), nn.ReLU(),
   268:             nn.Linear(256, 256), nn.ReLU(),
   269:             nn.Linear(256, action_dim), nn.Tanh(),
   270:         )
   271: 
   272:     def forward(self, state: torch.Tensor) -> torch.Tensor:
   273:         return self.max_action * self.net(state)
   274: 
   275:     @torch.no_grad()
   276:     def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
   277:         state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
   278:         return self(state).cpu().data.numpy().flatten()
   279: 
   280: 
   281: class Actor(nn.Module):
   282:     """Tanh-Gaussian stochastic policy. Default: 3 x 256 MLP.
   283:     Suitable for CQL, AWAC, Cal-QL style algorithms."""
   284: 
   285:     def __init__(self, state_dim: int, action_dim: int, max_action: float,
   286:                  orthogonal_init: bool = False):
   287:         super().__init__()
   288:         self.max_action = max_action
   289:         self.action_dim = action_dim
   290:         self.net = nn.Sequential(
   291:             nn.Linear(state_dim, 256), nn.ReLU(),
   292:             nn.Linear(256, 256), nn.ReLU(),
   293:             nn.Linear(256, 256), nn.ReLU(),
   294:             nn.Linear(256, 2 * action_dim),
   295:         )
   296:         init_module_weights(self.net, orthogonal_init)
   297:         self.log_std_min = -20.0
   298:         self.log_std_max = 2.0
   299: 
   300:     def _get_dist(self, state: torch.Tensor):
   301:         out = self.net(state)
   302:         mean, log_std = torch.split(out, self.action_dim, dim=-1)
   303:         log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
   304:         return TransformedDistribution(
   305:             Normal(mean, torch.exp(log_std)), TanhTransform(cache_size=1)
   306:         ), mean
   307: 
   308:     def forward(self, state: torch.Tensor, deterministic: bool = False):
   309:         dist, mean = self._get_dist(state)
   310:         action = torch.tanh(mean) if deterministic else dist.rsample()
   311:         log_prob = dist.log_prob(action).sum(-1)
   312:         return self.max_action * action, log_prob
   313: 
   314:     def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
   315:         """Log-probability of a dataset action under the current policy."""
   316:         dist, _ = self._get_dist(state)
   317:         action = torch.clamp(action / self.max_action, -1.0 + 1e-6, 1.0 - 1e-6)
   318:         return dist.log_prob(action).sum(-1)
   319: 
   320:     @torch.no_grad()
   321:     def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
   322:         state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
   323:         actions, _ = self(state, not self.training)
   324:         return actions.cpu().data.numpy().flatten()
   325: 
   326: 
   327: class Critic(nn.Module):
   328:     """Q-function Q(s, a). Default: 3 x 256 MLP."""
   329: 
   330:     def __init__(self, state_dim: int, action_dim: int, orthogonal_init: bool = False):
   331:         super().__init__()
   332:         self.net = nn.Sequential(
   333:             nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
   334:             nn.Linear(256, 256), nn.ReLU(),
   335:             nn.Linear(256, 256), nn.ReLU(),
   336:             nn.Linear(256, 1),
   337:         )
   338:         init_module_weights(self.net, orthogonal_init)
   339: 
   340:     def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
   341:         return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)
   342: 
   343: 
   344: class ValueFunction(nn.Module):
   345:     """State value function V(s). Default: 3 x 256 MLP."""
   346: 
   347:     def __init__(self, state_dim: int, orthogonal_init: bool = False):
   348:         super().__init__()
   349:         self.net = nn.Sequential(
   350:             nn.Linear(state_dim, 256), nn.ReLU(),
   351:             nn.Linear(256, 256), nn.ReLU(),
   352:             nn.Linear(256, 256), nn.ReLU(),
   353:             nn.Linear(256, 1),
   354:         )
   355:         init_module_weights(self.net, orthogonal_init)
   356: 
   357:     def forward(self, state: torch.Tensor) -> torch.Tensor:
   358:         return self.net(state).squeeze(-1)
   359: 
   360: 
   361: class OfflineOnlineAlgorithm:
   362:     """Offline-to-Online RL algorithm — implement your approach here.
   363: 
   364:     Goal: learn a policy from offline data (phase 1) and then improve it with
   365:     online interaction (phase 2) on Adroit dexterous manipulation tasks.
   366: 
   367:     Key challenges:
   368:     - Preventing Q-value collapse at the offline→online transition
   369:     - Avoiding catastrophic forgetting of offline-learned behaviors
   370:     - Handling distribution shift between offline and online data
   371:     - Balancing exploration vs exploitation during online fine-tuning
   372: 
   373:     The training loop calls:
   374:         trainer = OfflineOnlineAlgorithm(state_dim, action_dim, max_action, **kwargs)
   375: 
   376:         # Phase 1: Offline (1M steps)
   377:         for t in range(offline_iterations):
   378:             log_dict = trainer.train(batch, is_online=False)
   379: 
   380:         # Transition
   381:         trainer.on_online_start()
   382: 
   383:         # Phase 2: Online (1M steps)
   384:         for t in range(online_iterations):
   385:             action = trainer.select_action(state)
   386:             next_state, reward, done, info = env.step(action)
   387:             replay_buffer.add_transition(...)
   388:             log_dict = trainer.train(batch, is_online=True)
   389: 
   390:     You MUST set self.actor to an nn.Module that has an .act(state, device) method.
   391: 
   392:     Dataset access:
   393:         replay_buffer is a ReplayBuffer instance containing the offline dataset
   394:         (and later also online data). You can use it to compute dataset-level
   395:         statistics, e.g.:
   396:             replay_buffer._states[:replay_buffer._size]   — all states  (Tensor)
   397:             replay_buffer._actions[:replay_buffer._size]  — all actions (Tensor)
   398:             replay_buffer._rewards[:replay_buffer._size]  — all rewards (Tensor)
   399:             replay_buffer._next_actions[:replay_buffer._size] — next actions (Tensor)
   400:             replay_buffer._size                           — number of transitions
   401:         Note: during online phase, the buffer grows as new transitions are added.
   402: 
   403:     Available network classes (defined above, editable):
   404:         DeterministicActor  — deterministic tanh policy (for TD3+BC / SPOT approaches)
   405:         Actor               — stochastic Tanh-Gaussian policy (for CQL / AWAC / Cal-QL)
   406:         Critic              — Q(s, a) network
   407:         ValueFunction       — V(s) network
   408:     Available utilities (fixed): soft_update, init_module_weights
   409:     """
   410: 
   411:     def __init__(
   412:         self,
   413:         state_dim: int,
   414:         action_dim: int,
   415:         max_action: float,
   416:         replay_buffer: "ReplayBuffer" = None,
   417:         discount: float = 0.99,
   418:         tau: float = 5e-3,
   419:         actor_lr: float = 3e-4,
   420:         critic_lr: float = 3e-4,
   421:         device: str = "cuda",
   422:     ):
   423:         self.device = device
   424:         self.discount = discount
   425:         self.tau = tau
   426:         self.max_action = max_action
   427:         self.total_it = 0
   428:         # Full dataset buffer — use for computing global statistics.
   429:         # During online phase, this buffer grows as new transitions are added.
   430:         self.replay_buffer = replay_buffer
   431: 
   432:         # Build networks — modify or replace as needed
   433:         self.actor = Actor(state_dim, action_dim, max_action).to(device)
   434:         self.critic_1 = Critic(state_dim, action_dim).to(device)
   435:         self.critic_2 = Critic(state_dim, action_dim).to(device)
   436:         self.critic_1_target = deepcopy(self.critic_1)
   437:         self.critic_2_target = deepcopy(self.critic_2)
   438: 
   439:         self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
   440:         self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
   441:         self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
   442: 
   443:     def train(self, batch: TensorBatch, is_online: bool = False) -> Dict[str, float]:
   444:         """Update networks on one batch. Return a dict of scalar metrics for logging.
   445: 
   446:         batch = [states, actions, rewards, next_states, dones, next_actions]
   447:         (torch.Tensor, on device)
   448:         is_online = True during online fine-tuning phase
   449: 
   450:         next_actions is the action at the next timestep in the dataset
   451:         (useful for algorithms like ReBRAC that penalize deviations from
   452:         the data's next action). For online transitions, next_actions is
   453:         a placeholder (same as actions).
   454: 
   455:         TODO: implement your offline-to-online RL algorithm here.
   456:         """
   457:         self.total_it += 1
   458:         states, actions, rewards, next_states, dones, next_actions = batch
   459: 
   460:         # ── Placeholder: replace with your algorithm ──────────────────
   461:         log_dict: Dict[str, float] = {
   462:             "actor_loss": 0.0,
   463:             "critic_loss": 0.0,
   464:         }
   465:         return log_dict
   466: 
   467:     def select_action(self, state: np.ndarray) -> np.ndarray:
   468:         """Select action for online data collection. May add exploration noise."""
   469:         return self.actor.act(state, self.device)
   470: 
   471:     def on_online_start(self):
   472:         """Called once when transitioning from offline to online phase.
   473: 
   474:         Use this to reset optimizers, adjust hyperparameters, etc.
   475:         """
   476:         pass
   477: 
   478: 
   479: # =====================================================================
   480: # FIXED: Training loop (offline pretraining + online fine-tuning)
   481: # =====================================================================
   482: @pyrallis.wrap()
   483: def train(config: TrainConfig):
   484:     # Apply editable config overrides
   485:     _allowed_overrides = {
   486:         "normalize", "normalize_reward", "actor_lr", "critic_lr",
   487:         "tau", "expl_noise", "discount",
   488:     }
   489:     for _k, _v in CONFIG_OVERRIDES.items():
   490:         if _k not in _allowed_overrides:
   491:             raise ValueError(f"Unsupported CONFIG_OVERRIDES key: {_k}")
   492:         setattr(config, _k, _v)
   493: 
   494:     env = gym.make(config.env)
   495:     eval_env = gym.make(config.env)
   496: 
   497:     is_env_with_goal = config.env.startswith(ENVS_WITH_GOAL)
   498:     max_steps = env._max_episode_steps
   499: 
   500:     state_dim = env.observation_space.shape[0]

[truncated: showing at most 500 lines / 60000 bytes from CORL/algorithms/finetune/custom_finetune.py]
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


### `iql` baseline — editable region  [READ-ONLY — reference implementation]

In `CORL/algorithms/finetune/custom_finetune.py`:

```python
Lines 258–462:
   255: # expl_noise, discount.
   256: # Example: CONFIG_OVERRIDES = {"normalize": False, "actor_lr": 1e-3}
   257: # =====================================================================
   258: CONFIG_OVERRIDES: Dict[str, Any] = {}
   259: def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
   260:     return torch.mean(torch.abs(tau - (u < 0).float()) * u ** 2)
   261: 
   262: class DeterministicActor(nn.Module):
   263:     """Deterministic policy pi(s) = tanh(net(s)) * max_action.
   264:     Suitable for TD3+BC / SPOT style algorithms. Default: 2 x 256 MLP."""
   265: 
   266:     def __init__(self, state_dim: int, action_dim: int, max_action: float):
   267:         super().__init__()
   268:         self.max_action = max_action
   269:         self.net = nn.Sequential(
   270:             nn.Linear(state_dim, 256), nn.ReLU(),
   271:             nn.Linear(256, 256), nn.ReLU(),
   272:             nn.Linear(256, action_dim), nn.Tanh(),
   273:         )
   274: 
   275:     def forward(self, state: torch.Tensor) -> torch.Tensor:
   276:         return self.max_action * self.net(state)
   277: 
   278:     @torch.no_grad()
   279:     def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
   280:         state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
   281:         return self(state).cpu().data.numpy().flatten()
   282: 
   283: 
   284: class Actor(nn.Module):
   285:     """IQL GaussianPolicy — 2x256 MLP with Tanh output, state-independent log_std, Normal dist."""
   286: 
   287:     def __init__(self, state_dim: int, action_dim: int, max_action: float,
   288:                  hidden_dim: int = 256, n_hidden: int = 2, dropout: float = 0.1):
   289:         super().__init__()
   290:         dims = [state_dim] + [hidden_dim] * n_hidden + [action_dim]
   291:         layers = []
   292:         for i in range(len(dims) - 2):
   293:             layers.append(nn.Linear(dims[i], dims[i + 1]))
   294:             layers.append(nn.ReLU())
   295:             if dropout > 0.0:
   296:                 layers.append(nn.Dropout(dropout))
   297:         layers.append(nn.Linear(dims[-2], dims[-1]))
   298:         layers.append(nn.Tanh())
   299:         self.net = nn.Sequential(*layers)
   300:         self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
   301:         self.max_action = max_action
   302:         self._log_std_min = -20.0
   303:         self._log_std_max = 2.0
   304: 
   305:     def forward(self, obs: torch.Tensor) -> Normal:
   306:         mean = self.net(obs)
   307:         std = torch.exp(self.log_std.clamp(self._log_std_min, self._log_std_max))
   308:         return Normal(mean, std)
   309: 
   310:     @torch.no_grad()
   311:     def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
   312:         state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
   313:         dist = self(state)
   314:         action = dist.mean if not self.training else dist.sample()
   315:         action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
   316:         return action.cpu().data.numpy().flatten()
   317: 
   318: 
   319: class TwinQ(nn.Module):
   320:     """Twin Q-functions Q1(s,a), Q2(s,a). 2x256 MLPs, squeezed output."""
   321: 
   322:     def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
   323:         super().__init__()
   324:         dims = [state_dim + action_dim] + [hidden_dim] * n_hidden + [1]
   325: 
   326:         def _build_mlp():
   327:             layers = []
   328:             for i in range(len(dims) - 2):
   329:                 layers.append(nn.Linear(dims[i], dims[i + 1]))
   330:                 layers.append(nn.ReLU())
   331:             layers.append(nn.Linear(dims[-2], dims[-1]))
   332:             return nn.Sequential(*layers)
   333: 
   334:         self.q1 = _build_mlp()
   335:         self.q2 = _build_mlp()
   336: 
   337:     def both(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
   338:         sa = torch.cat([state, action], dim=1)
   339:         return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)
   340: 
   341:     def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
   342:         return torch.min(*self.both(state, action))
   343: 
   344: class ValueFunction(nn.Module):
   345:     """State value function V(s). 2x256 MLP, squeezed output."""
   346: 
   347:     def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
   348:         super().__init__()
   349:         dims = [state_dim] + [hidden_dim] * n_hidden + [1]
   350:         layers = []
   351:         for i in range(len(dims) - 2):
   352:             layers.append(nn.Linear(dims[i], dims[i + 1]))
   353:             layers.append(nn.ReLU())
   354:         layers.append(nn.Linear(dims[-2], dims[-1]))
   355:         self.v = nn.Sequential(*layers)
   356: 
   357:     def forward(self, state: torch.Tensor) -> torch.Tensor:
   358:         return self.v(state).squeeze(-1)
   359: 
   360: class OfflineOnlineAlgorithm:
   361:     """IQL — Implicit Q-Learning for offline-to-online RL."""
   362: 
   363:     def __init__(
   364:         self,
   365:         state_dim: int,
   366:         action_dim: int,
   367:         max_action: float,
   368:         replay_buffer=None,
   369:         discount: float = 0.99,
   370:         tau: float = 5e-3,
   371:         actor_lr: float = 3e-4,
   372:         critic_lr: float = 3e-4,
   373:         device: str = "cuda",
   374:     ):
   375:         self.device = device
   376:         self.discount = discount
   377:         self.tau = tau
   378:         self.max_action = max_action
   379:         self.total_it = 0
   380: 
   381:         # IQL hyperparameters (match CORL reference)
   382:         self.iql_tau = 0.8       # expectile for asymmetric V loss (CORL hammer/pen-cloned config)
   383:         self.beta = 3.0          # inverse temperature for advantage weighting
   384:         self.exp_adv_max = 100.0
   385: 
   386:         # Actor (GaussianPolicy-style)
   387:         self.actor = Actor(state_dim, action_dim, max_action).to(device)
   388:         self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
   389:         self.actor_lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
   390:             self.actor_optimizer, T_max=int(1e6)
   391:         )
   392: 
   393:         # Twin Q-network + target
   394:         self.qf = TwinQ(state_dim, action_dim).to(device)
   395:         self.q_target = deepcopy(self.qf).requires_grad_(False).to(device)
   396:         self.q_optimizer = torch.optim.Adam(self.qf.parameters(), lr=critic_lr)
   397: 
   398:         # Value function
   399:         self.vf = ValueFunction(state_dim).to(device)
   400:         self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=critic_lr)
   401: 
   402:     def train(self, batch: TensorBatch, is_online: bool = False) -> Dict[str, float]:
   403:         self.total_it += 1
   404:         states, actions, rewards, next_states, dones, *_ = batch
   405:         rewards = rewards.squeeze(dim=-1)
   406:         dones = dones.squeeze(dim=-1)
   407:         log_dict: Dict[str, float] = {}
   408: 
   409:         # V(s) update — expectile regression
   410:         with torch.no_grad():
   411:             target_q = self.q_target(states, actions)
   412:         v = self.vf(states)
   413:         adv = target_q - v
   414:         v_loss = asymmetric_l2_loss(adv, self.iql_tau)
   415:         log_dict["value_loss"] = v_loss.item()
   416: 
   417:         self.v_optimizer.zero_grad()
   418:         v_loss.backward()
   419:         self.v_optimizer.step()
   420: 
   421:         # Q(s,a) update — standard Bellman with target V
   422:         with torch.no_grad():
   423:             next_v = self.vf(next_states)
   424:         targets = rewards + (1.0 - dones) * self.discount * next_v.detach()
   425:         qs = self.qf.both(states, actions)
   426:         q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
   427:         log_dict["q_loss"] = q_loss.item()
   428: 
   429:         self.q_optimizer.zero_grad()
   430:         q_loss.backward()
   431:         self.q_optimizer.step()
   432: 
   433:         # Target Q update
   434:         soft_update(self.q_target, self.qf, self.tau)
   435: 
   436:         # Policy update — advantage-weighted regression
   437:         exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=self.exp_adv_max)
   438:         policy_out = self.actor(states)
   439:         if isinstance(policy_out, torch.distributions.Distribution):
   440:             bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
   441:         elif torch.is_tensor(policy_out):
   442:             if policy_out.shape != actions.shape:
   443:                 raise RuntimeError("Actions shape mismatch")
   444:             bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
   445:         else:
   446:             raise NotImplementedError
   447:         policy_loss = torch.mean(exp_adv * bc_losses)
   448:         log_dict["actor_loss"] = policy_loss.item()
   449: 
   450:         self.actor_optimizer.zero_grad()
   451:         policy_loss.backward()
   452:         self.actor_optimizer.step()
   453:         self.actor_lr_schedule.step()
   454: 
   455:         return log_dict
   456: 
   457:     def select_action(self, state: np.ndarray) -> np.ndarray:
   458:         return self.actor.act(state, self.device)
   459: 
   460:     def on_online_start(self):
   461:         # IQL needs no special handling at offline-to-online transition
   462:         pass
   463: 
   464: # =====================================================================
   465: # FIXED: Training loop (offline pretraining + online fine-tuning)
```

### `awac` baseline — editable region  [READ-ONLY — reference implementation]

In `CORL/algorithms/finetune/custom_finetune.py`:

```python
Lines 258–460:
   255: # expl_noise, discount.
   256: # Example: CONFIG_OVERRIDES = {"normalize": False, "actor_lr": 1e-3}
   257: # =====================================================================
   258: CONFIG_OVERRIDES: Dict[str, Any] = {}
   259: class DeterministicActor(nn.Module):
   260:     """Deterministic policy pi(s) = tanh(net(s)) * max_action.
   261:     Suitable for TD3+BC / SPOT style algorithms. Default: 2 x 256 MLP."""
   262: 
   263:     def __init__(self, state_dim: int, action_dim: int, max_action: float):
   264:         super().__init__()
   265:         self.max_action = max_action
   266:         self.net = nn.Sequential(
   267:             nn.Linear(state_dim, 256), nn.ReLU(),
   268:             nn.Linear(256, 256), nn.ReLU(),
   269:             nn.Linear(256, action_dim), nn.Tanh(),
   270:         )
   271: 
   272:     def forward(self, state: torch.Tensor) -> torch.Tensor:
   273:         return self.max_action * self.net(state)
   274: 
   275:     @torch.no_grad()
   276:     def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
   277:         state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
   278:         return self(state).cpu().data.numpy().flatten()
   279: 
   280: 
   281: class Actor(nn.Module):
   282:     """AWAC GaussianPolicy — 3x256 MLP, state-independent log_std, Normal + clamp."""
   283: 
   284:     def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
   285:                  min_log_std: float = -20.0, max_log_std: float = 2.0):
   286:         super().__init__()
   287:         self._mlp = nn.Sequential(
   288:             nn.Linear(state_dim, hidden_dim),
   289:             nn.ReLU(),
   290:             nn.Linear(hidden_dim, hidden_dim),
   291:             nn.ReLU(),
   292:             nn.Linear(hidden_dim, hidden_dim),
   293:             nn.ReLU(),
   294:             nn.Linear(hidden_dim, action_dim),
   295:         )
   296:         self._log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
   297:         self._min_log_std = min_log_std
   298:         self._max_log_std = max_log_std
   299: 
   300:     def _get_policy(self, state: torch.Tensor):
   301:         mean = self._mlp(state)
   302:         log_std = self._log_std.clamp(self._min_log_std, self._max_log_std)
   303:         return torch.distributions.Normal(mean, log_std.exp())
   304: 
   305:     def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
   306:         policy = self._get_policy(state)
   307:         return policy.log_prob(action).sum(-1, keepdim=True)
   308: 
   309:     def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
   310:         policy = self._get_policy(state)
   311:         action = policy.rsample()
   312:         action.clamp_(-1.0, 1.0)
   313:         log_prob = policy.log_prob(action).sum(-1, keepdim=True)
   314:         return action, log_prob
   315: 
   316:     def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
   317:         state_t = torch.tensor(state[None], dtype=torch.float32, device=device)
   318:         policy = self._get_policy(state_t)
   319:         if self._mlp.training:
   320:             action_t = policy.sample()
   321:         else:
   322:             action_t = policy.mean
   323:         return action_t[0].cpu().numpy()
   324: 
   325: 
   326: class Critic(nn.Module):
   327:     """Q-function Q(s, a). 3x256 MLP, returns (batch, 1)."""
   328: 
   329:     def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
   330:         super().__init__()
   331:         self._mlp = nn.Sequential(
   332:             nn.Linear(state_dim + action_dim, hidden_dim),
   333:             nn.ReLU(),
   334:             nn.Linear(hidden_dim, hidden_dim),
   335:             nn.ReLU(),
   336:             nn.Linear(hidden_dim, hidden_dim),
   337:             nn.ReLU(),
   338:             nn.Linear(hidden_dim, 1),
   339:         )
   340: 
   341:     def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
   342:         return self._mlp(torch.cat([state, action], dim=-1))
   343: 
   344: class ValueFunction(nn.Module):
   345:     """State value function V(s). Default: 3 x 256 MLP."""
   346: 
   347:     def __init__(self, state_dim: int, orthogonal_init: bool = False):
   348:         super().__init__()
   349:         self.net = nn.Sequential(
   350:             nn.Linear(state_dim, 256), nn.ReLU(),
   351:             nn.Linear(256, 256), nn.ReLU(),
   352:             nn.Linear(256, 256), nn.ReLU(),
   353:             nn.Linear(256, 1),
   354:         )
   355:         init_module_weights(self.net, orthogonal_init)
   356: 
   357:     def forward(self, state: torch.Tensor) -> torch.Tensor:
   358:         return self.net(state).squeeze(-1)
   359: 
   360: 
   361: class OfflineOnlineAlgorithm:
   362:     """AWAC — Advantage Weighted Actor-Critic for offline-to-online RL."""
   363: 
   364:     def __init__(
   365:         self,
   366:         state_dim: int,
   367:         action_dim: int,
   368:         max_action: float,
   369:         replay_buffer=None,
   370:         discount: float = 0.99,
   371:         tau: float = 5e-3,
   372:         actor_lr: float = 3e-4,
   373:         critic_lr: float = 3e-4,
   374:         device: str = "cuda",
   375:     ):
   376:         self.device = device
   377:         self.discount = discount
   378:         self.tau = tau
   379:         self.max_action = max_action
   380:         self.total_it = 0
   381: 
   382:         # AWAC hyperparameters (match CORL reference: awac_lambda=0.1)
   383:         self.awac_lambda = 0.1
   384:         self.exp_adv_max = 100.0
   385: 
   386:         # Actor (GaussianPolicy-style with state-independent log_std)
   387:         self.actor = Actor(state_dim, action_dim, 256).to(device)
   388:         self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
   389: 
   390:         # Twin critics + targets (SEPARATE optimizers)
   391:         self.critic_1 = Critic(state_dim, action_dim, 256).to(device)
   392:         self.critic_2 = Critic(state_dim, action_dim, 256).to(device)
   393:         self.target_critic_1 = deepcopy(self.critic_1)
   394:         self.target_critic_2 = deepcopy(self.critic_2)
   395:         self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
   396:         self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
   397: 
   398:     def train(self, batch: TensorBatch, is_online: bool = False) -> Dict[str, float]:
   399:         self.total_it += 1
   400:         states, actions, rewards, next_states, dones, *_ = batch
   401:         log_dict: Dict[str, float] = {}
   402: 
   403:         # Critic update
   404:         with torch.no_grad():
   405:             next_actions, _ = self.actor(next_states)
   406:             q_next = torch.min(
   407:                 self.target_critic_1(next_states, next_actions),
   408:                 self.target_critic_2(next_states, next_actions),
   409:             )
   410:             q_target = rewards + self.discount * (1.0 - dones) * q_next
   411: 
   412:         q1 = self.critic_1(states, actions)
   413:         q2 = self.critic_2(states, actions)
   414:         q1_loss = F.mse_loss(q1, q_target)
   415:         q2_loss = F.mse_loss(q2, q_target)
   416:         critic_loss = q1_loss + q2_loss
   417:         log_dict["critic_loss"] = critic_loss.item()
   418: 
   419:         self.critic_1_optimizer.zero_grad()
   420:         self.critic_2_optimizer.zero_grad()
   421:         critic_loss.backward()
   422:         self.critic_1_optimizer.step()
   423:         self.critic_2_optimizer.step()
   424: 
   425:         # Actor update (advantage-weighted)
   426:         with torch.no_grad():
   427:             pi_action, _ = self.actor(states)
   428:             v = torch.min(
   429:                 self.critic_1(states, pi_action),
   430:                 self.critic_2(states, pi_action),
   431:             )
   432:             q = torch.min(
   433:                 self.critic_1(states, actions),
   434:                 self.critic_2(states, actions),
   435:             )
   436:             adv = q - v
   437:             weights = torch.clamp_max(
   438:                 torch.exp(adv / self.awac_lambda), self.exp_adv_max
   439:             )
   440: 
   441:         action_log_prob = self.actor.log_prob(states, actions)
   442:         actor_loss = (-action_log_prob * weights).mean()
   443:         log_dict["actor_loss"] = actor_loss.item()
   444: 
   445:         self.actor_optimizer.zero_grad()
   446:         actor_loss.backward()
   447:         self.actor_optimizer.step()
   448: 
   449:         # Target update
   450:         soft_update(self.target_critic_1, self.critic_1, self.tau)
   451:         soft_update(self.target_critic_2, self.critic_2, self.tau)
   452: 
   453:         return log_dict
   454: 
   455:     def select_action(self, state: np.ndarray) -> np.ndarray:
   456:         return self.actor.act(state, self.device)
   457: 
   458:     def on_online_start(self):
   459:         # AWAC needs no special handling at transition
   460:         pass
   461: 
   462: # =====================================================================
   463: # FIXED: Training loop (offline pretraining + online fine-tuning)
```

### `spot` baseline — editable region  [READ-ONLY — reference implementation]

In `CORL/algorithms/finetune/custom_finetune.py`:

```python
Lines 258–606:
   255: # expl_noise, discount.
   256: # Example: CONFIG_OVERRIDES = {"normalize": False, "actor_lr": 1e-3}
   257: # =====================================================================
   258: CONFIG_OVERRIDES: Dict[str, Any] = {"normalize": False}
   259: class VAE(nn.Module):
   260:     """Variational Auto-Encoder for SPOT support constraint."""
   261: 
   262:     def __init__(self, state_dim: int, action_dim: int, latent_dim: int,
   263:                  max_action: float, hidden_dim: int = 750):
   264:         super().__init__()
   265:         self.encoder_shared = nn.Sequential(
   266:             nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
   267:             nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
   268:         )
   269:         self.mean = nn.Linear(hidden_dim, latent_dim)
   270:         self.log_std = nn.Linear(hidden_dim, latent_dim)
   271:         self.decoder = nn.Sequential(
   272:             nn.Linear(state_dim + latent_dim, hidden_dim), nn.ReLU(),
   273:             nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
   274:             nn.Linear(hidden_dim, action_dim), nn.Tanh(),
   275:         )
   276:         self.max_action = max_action
   277:         self.latent_dim = latent_dim
   278: 
   279:     def forward(self, state, action):
   280:         mean, std = self.encode(state, action)
   281:         z = mean + std * torch.randn_like(std)
   282:         u = self.decode(state, z)
   283:         return u, mean, std
   284: 
   285:     def encode(self, state, action):
   286:         z = self.encoder_shared(torch.cat([state, action], -1))
   287:         mean = self.mean(z)
   288:         log_std = self.log_std(z).clamp(-4, 15)
   289:         std = torch.exp(log_std)
   290:         return mean, std
   291: 
   292:     def decode(self, state, z):
   293:         return self.max_action * self.decoder(torch.cat([state, z], -1))
   294: 
   295: class DeterministicActor(nn.Module):
   296:     """Deterministic policy pi(s) = tanh(net(s)) * max_action.
   297:     Suitable for TD3+BC / SPOT style algorithms. Default: 2 x 256 MLP."""
   298: 
   299:     def __init__(self, state_dim: int, action_dim: int, max_action: float):
   300:         super().__init__()
   301:         self.max_action = max_action
   302:         self.net = nn.Sequential(
   303:             nn.Linear(state_dim, 256), nn.ReLU(),
   304:             nn.Linear(256, 256), nn.ReLU(),
   305:             nn.Linear(256, action_dim), nn.Tanh(),
   306:         )
   307: 
   308:     def forward(self, state: torch.Tensor) -> torch.Tensor:
   309:         return self.max_action * self.net(state)
   310: 
   311:     @torch.no_grad()
   312:     def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
   313:         state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
   314:         return self(state).cpu().data.numpy().flatten()
   315: 
   316: 
   317: class Actor(nn.Module):
   318:     """Tanh-Gaussian stochastic policy. Default: 3 x 256 MLP.
   319:     Suitable for CQL, AWAC, Cal-QL style algorithms."""
   320: 
   321:     def __init__(self, state_dim: int, action_dim: int, max_action: float,
   322:                  orthogonal_init: bool = False):
   323:         super().__init__()
   324:         self.max_action = max_action
   325:         self.action_dim = action_dim
   326:         self.net = nn.Sequential(
   327:             nn.Linear(state_dim, 256), nn.ReLU(),
   328:             nn.Linear(256, 256), nn.ReLU(),
   329:             nn.Linear(256, 256), nn.ReLU(),
   330:             nn.Linear(256, 2 * action_dim),
   331:         )
   332:         init_module_weights(self.net, orthogonal_init)
   333:         self.log_std_min = -20.0
   334:         self.log_std_max = 2.0
   335: 
   336:     def _get_dist(self, state: torch.Tensor):
   337:         out = self.net(state)
   338:         mean, log_std = torch.split(out, self.action_dim, dim=-1)
   339:         log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
   340:         return TransformedDistribution(
   341:             Normal(mean, torch.exp(log_std)), TanhTransform(cache_size=1)
   342:         ), mean
   343: 
   344:     def forward(self, state: torch.Tensor, deterministic: bool = False):
   345:         dist, mean = self._get_dist(state)
   346:         action = torch.tanh(mean) if deterministic else dist.rsample()
   347:         log_prob = dist.log_prob(action).sum(-1)
   348:         return self.max_action * action, log_prob
   349: 
   350:     def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
   351:         """Log-probability of a dataset action under the current policy."""
   352:         dist, _ = self._get_dist(state)
   353:         action = torch.clamp(action / self.max_action, -1.0 + 1e-6, 1.0 - 1e-6)
   354:         return dist.log_prob(action).sum(-1)
   355: 
   356:     @torch.no_grad()
   357:     def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
   358:         state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
   359:         actions, _ = self(state, not self.training)
   360:         return actions.cpu().data.numpy().flatten()
   361: 
   362: 
   363: class Critic(nn.Module):
   364:     """Q-function Q(s, a). 2x256 MLP, returns (batch, 1)."""
   365: 
   366:     def __init__(self, state_dim: int, action_dim: int, orthogonal_init: bool = False):
   367:         super().__init__()
   368:         self.net = nn.Sequential(
   369:             nn.Linear(state_dim + action_dim, 256),
   370:             nn.ReLU(),
   371:             nn.Linear(256, 256),
   372:             nn.ReLU(),
   373:             nn.Linear(256, 1),
   374:         )
   375: 
   376:     def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
   377:         return self.net(torch.cat([state, action], dim=-1))
   378: 
   379: class ValueFunction(nn.Module):
   380:     """State value function V(s). Default: 3 x 256 MLP."""
   381: 
   382:     def __init__(self, state_dim: int, orthogonal_init: bool = False):
   383:         super().__init__()
   384:         self.net = nn.Sequential(
   385:             nn.Linear(state_dim, 256), nn.ReLU(),
   386:             nn.Linear(256, 256), nn.ReLU(),
   387:             nn.Linear(256, 256), nn.ReLU(),
   388:             nn.Linear(256, 1),
   389:         )
   390:         init_module_weights(self.net, orthogonal_init)
   391: 
   392:     def forward(self, state: torch.Tensor) -> torch.Tensor:
   393:         return self.net(state).squeeze(-1)
   394: 
   395: 
   396: class OfflineOnlineAlgorithm:
   397:     """SPOT — Support constraint Policy Optimization via online Training (TD3 + VAE)."""
   398: 
   399:     def __init__(
   400:         self,
   401:         state_dim: int,
   402:         action_dim: int,
   403:         max_action: float,
   404:         replay_buffer=None,
   405:         discount: float = 0.99,
   406:         tau: float = 5e-3,
   407:         actor_lr: float = 3e-4,
   408:         critic_lr: float = 3e-4,
   409:         device: str = "cuda",
   410:     ):
   411:         self.device = device
   412:         self.discount = discount
   413:         self.online_discount = 0.99
   414:         self.tau = tau
   415:         self.max_action = max_action
   416:         self.total_it = 0
   417:         self.replay_buffer = replay_buffer
   418: 
   419:         # SPOT hyperparameters
   420:         self.policy_noise = 0.2 * max_action
   421:         self.noise_clip = 0.5 * max_action
   422:         self.policy_freq = 2
   423:         self.expl_noise = 0.1
   424:         self.beta = 0.5
   425:         self.lambd = 1.0
   426:         self.lambd_cool = True
   427:         self.lambd_end = 0.5
   428:         self.num_samples = 1
   429:         self.is_online = False
   430:         self.online_it = 0
   431:         self.max_online_steps = int(1e6)
   432:         self.vae_iterations = 100_000
   433:         self._actor_lr = 1e-4
   434:         self._critic_lr = critic_lr
   435: 
   436:         # VAE for support constraint (pretrained before offline policy training)
   437:         latent_dim = 2 * action_dim
   438:         self.vae = VAE(state_dim, action_dim, latent_dim, max_action, hidden_dim=750).to(device)
   439:         self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3)
   440:         self._vae_trained = False
   441: 
   442:         # Actor (deterministic) + target — init head weights small (reference: 0.001)
   443:         self.actor = DeterministicActor(state_dim, action_dim, max_action).to(device)
   444:         _actor_head = self.actor.net[-2]  # Linear before Tanh
   445:         _actor_head.weight.data.uniform_(-0.001, 0.001)
   446:         _actor_head.bias.data.uniform_(-0.001, 0.001)
   447:         self.actor_target = deepcopy(self.actor)
   448:         self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self._actor_lr)
   449: 
   450:         # Twin critics (2x256, unsqueezed) + targets — init head weights small (reference: 0.003)
   451:         self.critic_1 = Critic(state_dim, action_dim).to(device)
   452:         _c1_head = self.critic_1.net[-1]  # Last Linear (output layer)
   453:         _c1_head.weight.data.uniform_(-0.003, 0.003)
   454:         _c1_head.bias.data.uniform_(-0.003, 0.003)
   455:         self.critic_1_target = deepcopy(self.critic_1)
   456:         self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=self._critic_lr)
   457: 
   458:         self.critic_2 = Critic(state_dim, action_dim).to(device)
   459:         _c2_head = self.critic_2.net[-1]
   460:         _c2_head.weight.data.uniform_(-0.003, 0.003)
   461:         _c2_head.bias.data.uniform_(-0.003, 0.003)
   462:         self.critic_2_target = deepcopy(self.critic_2)
   463:         self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=self._critic_lr)
   464: 
   465:     def _elbo_loss(self, state, action):
   466:         """ELBO loss for VAE support constraint (neg log beta)."""
   467:         mean, std = self.vae.encode(state, action)
   468:         N = self.num_samples
   469:         mean_s = mean.repeat(N, 1, 1).permute(1, 0, 2)
   470:         std_s = std.repeat(N, 1, 1).permute(1, 0, 2)
   471:         z = mean_s + std_s * torch.randn_like(std_s)
   472:         state_r = state.repeat(N, 1, 1).permute(1, 0, 2)
   473:         action_r = action.repeat(N, 1, 1).permute(1, 0, 2)
   474:         u = self.vae.decode(state_r, z)
   475:         recon_loss = ((u - action_r) ** 2).mean(dim=(1, 2))
   476:         KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean(-1)
   477:         return recon_loss + self.beta * KL_loss
   478: 
   479:     def _vae_train_step(self, batch):
   480:         """One VAE training step."""
   481:         state, action, *_ = batch
   482:         recon, mean, std = self.vae(state, action)
   483:         recon_loss = F.mse_loss(recon, action)
   484:         KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
   485:         vae_loss = recon_loss + self.beta * KL_loss
   486:         self.vae_optimizer.zero_grad()
   487:         vae_loss.backward()
   488:         self.vae_optimizer.step()
   489:         return {"vae_loss": vae_loss.item(), "vae_recon": recon_loss.item()}
   490: 
   491:     def pretrain(self, replay_buffer, batch_size: int) -> Dict[str, float]:
   492:         """Train the VAE before the fixed offline-to-online loop starts."""
   493:         print(f"Pretraining SPOT VAE for {self.vae_iterations} steps")
   494:         log_dict: Dict[str, float] = {}
   495:         self.vae.train()
   496:         for t in range(self.vae_iterations):
   497:             batch = replay_buffer.sample(batch_size)
   498:             batch = [b.to(self.device) for b in batch]
   499:             log_dict = self._vae_train_step(batch)
   500:             if (t + 1) % 1000 == 0:
   501:                 metrics_str = " ".join(
   502:                     f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
   503:                     for k, v in log_dict.items()
   504:                 )
   505:                 print(f"TRAIN_METRICS step=vae_{t+1} {metrics_str}", flush=True)
   506:         self.vae.eval()
   507:         self._vae_trained = True
   508:         return log_dict
   509: 
   510:     def train(self, batch: TensorBatch, is_online: bool = False) -> Dict[str, float]:
   511:         self.total_it += 1
   512:         if not self._vae_trained:
   513:             if self.replay_buffer is not None:
   514:                 self.pretrain(self.replay_buffer, batch[0].shape[0])
   515:             else:
   516:                 self.vae.eval()
   517:                 self._vae_trained = True
   518: 
   519:         if is_online:
   520:             self.online_it += 1
   521:         state, action, reward, next_state, done, *_ = batch
   522:         not_done = 1 - done
   523:         log_dict: Dict[str, float] = {}
   524: 
   525:         # Critic update
   526:         with torch.no_grad():
   527:             noise = (torch.randn_like(action) * self.policy_noise).clamp(
   528:                 -self.noise_clip, self.noise_clip
   529:             )
   530:             next_action = (self.actor_target(next_state) + noise).clamp(
   531:                 -self.max_action, self.max_action
   532:             )
   533:             target_q1 = self.critic_1_target(next_state, next_action)
   534:             target_q2 = self.critic_2_target(next_state, next_action)
   535:             target_q = torch.min(target_q1, target_q2)
   536:             target_q = reward + not_done * self.discount * target_q
   537: 
   538:         current_q1 = self.critic_1(state, action)
   539:         current_q2 = self.critic_2(state, action)
   540:         critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
   541:         log_dict["critic_loss"] = critic_loss.item()
   542: 
   543:         self.critic_1_optimizer.zero_grad()
   544:         self.critic_2_optimizer.zero_grad()
   545:         critic_loss.backward()
   546:         self.critic_1_optimizer.step()
   547:         self.critic_2_optimizer.step()
   548: 
   549:         # Delayed actor updates with VAE support constraint
   550:         if self.total_it % self.policy_freq == 0:
   551:             pi = self.actor(state)
   552:             q = self.critic_1(state, pi)
   553: 
   554:             # VAE support constraint (neg log beta)
   555:             neg_log_beta = self._elbo_loss(state, pi)
   556: 
   557:             # Lambda cooling
   558:             if self.lambd_cool:
   559:                 lambd = self.lambd * max(
   560:                     self.lambd_end, (1.0 - self.online_it / self.max_online_steps)
   561:                 )
   562:             else:
   563:                 lambd = self.lambd
   564: 
   565:             # Q-value normalization
   566:             norm_q = 1.0 / q.abs().mean().detach()
   567: 
   568:             actor_loss = -norm_q * q.mean() + lambd * neg_log_beta.mean()
   569:             log_dict["actor_loss"] = actor_loss.item()
   570:             log_dict["lambd"] = lambd
   571: 
   572:             self.actor_optimizer.zero_grad()
   573:             actor_loss.backward()
   574:             self.actor_optimizer.step()
   575: 
   576:             soft_update(self.critic_1_target, self.critic_1, self.tau)
   577:             soft_update(self.critic_2_target, self.critic_2, self.tau)
   578:             soft_update(self.actor_target, self.actor, self.tau)
   579: 
   580:         return log_dict
   581: 
   582:     def select_action(self, state: np.ndarray) -> np.ndarray:
   583:         with torch.no_grad():
   584:             state_t = torch.tensor(
   585:                 state.reshape(1, -1), device=self.device, dtype=torch.float32
   586:             )
   587:             action = self.actor(state_t)
   588:             noise = (torch.randn_like(action) * self.expl_noise).clamp(
   589:                 -self.noise_clip, self.noise_clip
   590:             )
   591:             action = (action + noise).clamp(-self.max_action, self.max_action)
   592:         return action.cpu().data.numpy().flatten()
   593: 
   594:     def on_online_start(self):
   595:         self.is_online = True
   596:         self.discount = self.online_discount
   597:         # Reset optimizers at transition
   598:         self.actor_optimizer = torch.optim.Adam(
   599:             self.actor.parameters(), lr=self._actor_lr
   600:         )
   601:         self.critic_1_optimizer = torch.optim.Adam(
   602:             self.critic_1.parameters(), lr=self._critic_lr
   603:         )
   604:         self.critic_2_optimizer = torch.optim.Adam(
   605:             self.critic_2.parameters(), lr=self._critic_lr
   606:         )
   607: 
   608: # =====================================================================
   609: # FIXED: Training loop (offline pretraining + online fine-tuning)
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
