# MLS-Bench: rl-offline-continuous

# Offline RL: Q-Value Overestimation Suppression in Continuous Control

## Research Question
Design and implement an offline RL algorithm that suppresses Q-value
overestimation while learning useful policies from a static dataset.
Your code goes in `custom.py`. Several reference implementations are
provided as read-only `*.edit.py` baselines.

## Background
In offline RL the agent cannot collect new transitions, so standard
bootstrapped Q-learning tends to overestimate values for
out-of-distribution actions, which then drive the policy away from the
data and degrade performance. Different mechanisms — conservative value
penalties, behavior regularization, expectile-style value functions,
ensemble pessimism — trade off in-distribution exploitation against
out-of-distribution caution.

Reference baselines spanning the design space:
- **ReBRAC** — Tarasov et al., "Revisiting the Minimalist Approach to
  Offline Reinforcement Learning" (arXiv:2305.09836, NeurIPS 2023).
  Decoupled actor and critic BC penalties on top of TD3+BC.
- **TD3+BC** — Fujimoto and Gu, "A Minimalist Approach to Offline
  Reinforcement Learning" (arXiv:2106.06860, NeurIPS 2021). TD3
  augmented with a normalized BC term in the actor loss with default
  coefficient `alpha = 2.5`.
- **IQL** — Kostrikov et al., "Offline Reinforcement Learning with
  Implicit Q-Learning" (arXiv:2110.06169, ICLR 2022). Expectile
  regression with default `tau = 0.7` and advantage-weighted policy
  extraction temperature `beta = 3.0` for D4RL MuJoCo.

## Constraints
- **Network dimensions are fixed at 256.** All MLP hidden layers must
  use 256 units. A `_mlp()` factory function is provided in the FIXED
  section for convenience. You may define custom network classes but
  hidden widths must remain 256.
- **Total parameter count is enforced.** The training loop checks that
  total trainable parameters do not exceed 1.2x the largest baseline
  architecture, so the contribution must be algorithmic (losses,
  regularization, target construction, policy extraction) rather than
  capacity.
- Do **not** simply copy a reference implementation with minor changes.

## Evaluation
Trained and evaluated on D4RL MuJoCo continuous-control datasets
including HalfCheetah, Hopper and Walker2d using `medium-v2` data.
Metric: D4RL normalized score (0 = random, 100 = expert), averaged over
evaluation rollouts. Higher is better. Strong methods should generalize
across the locomotion datasets rather than relying on dataset-specific
quirks.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/CORL/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `CORL/algorithms/offline/custom.py`
- editable lines **193–397**




## Readable Context


### `CORL/algorithms/offline/custom.py`  [EDITABLE — lines 193–397 only]

```python
     1: # Custom offline RL algorithm for MLS-Bench
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
    31:     env: str = "halfcheetah-medium-v2"
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
    44:     normalize: bool = True
    45:     orthogonal_init: bool = True
    46:     project: str = "CORL"
    47:     group: str = "custom-D4RL"
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
   133:         # Compute next_actions: action at the next timestep in the dataset.
   134:         # At episode boundaries (terminal=1) or the last transition, use the current action.
   135:         next_actions = np.concatenate([data["actions"][1:], data["actions"][-1:]], axis=0)
   136:         terminals = data["terminals"].astype(bool)
   137:         next_actions[terminals] = data["actions"][terminals]
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
   177:     """FIXED MLP factory -- hidden width is locked at 256. Do NOT modify."""
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
   189: # =====================================================================
   190: # EDITABLE: Network definitions and OfflineAlgorithm
   191: #
   192: # CONSTRAINTS:
   193: # - Total trainable parameter count is soft-capped.
   194: # - Total parameter count is checked at runtime and must not exceed
   195: #   1.2x the largest baseline. Focus on algorithmic improvements, not
   196: #   network capacity.
   197: #
   198: # CONFIG_OVERRIDES: override method-specific TrainConfig fields here.
   199: # Allowed keys: normalize, normalize_reward, actor_lr, critic_lr, tau, discount.
   200: # Example: CONFIG_OVERRIDES = {"normalize": False, "actor_lr": 1e-3}
   201: # =====================================================================
   202: CONFIG_OVERRIDES: Dict[str, Any] = {}
   203: 
   204: 
   205: class DeterministicActor(nn.Module):
   206:     """Deterministic policy pi(s) = tanh(net(s)) * max_action.
   207:     Suitable for BC, TD3+BC style algorithms. Default: 2 x 256 MLP."""
   208: 
   209:     def __init__(self, state_dim: int, action_dim: int, max_action: float):
   210:         super().__init__()
   211:         self.max_action = max_action
   212:         self.net = nn.Sequential(
   213:             nn.Linear(state_dim, 256), nn.ReLU(),
   214:             nn.Linear(256, 256), nn.ReLU(),
   215:             nn.Linear(256, action_dim), nn.Tanh(),
   216:         )
   217: 
   218:     def forward(self, state: torch.Tensor) -> torch.Tensor:
   219:         return self.max_action * self.net(state)
   220: 
   221:     @torch.no_grad()
   222:     def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
   223:         state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
   224:         return self(state).cpu().data.numpy().flatten()
   225: 
   226: 
   227: class Actor(nn.Module):
   228:     """Tanh-Gaussian stochastic policy. Default: 3 x 256 MLP.
   229:     Suitable for CQL, IQL style algorithms."""
   230: 
   231:     def __init__(self, state_dim: int, action_dim: int, max_action: float,
   232:                  orthogonal_init: bool = False):
   233:         super().__init__()
   234:         self.max_action = max_action
   235:         self.action_dim = action_dim
   236:         self.net = nn.Sequential(
   237:             nn.Linear(state_dim, 256), nn.ReLU(),
   238:             nn.Linear(256, 256), nn.ReLU(),
   239:             nn.Linear(256, 256), nn.ReLU(),
   240:             nn.Linear(256, 2 * action_dim),
   241:         )
   242:         init_module_weights(self.net, orthogonal_init)
   243:         self.log_std_min = -20.0
   244:         self.log_std_max = 2.0
   245: 
   246:     def _get_dist(self, state: torch.Tensor):
   247:         out = self.net(state)
   248:         mean, log_std = torch.split(out, self.action_dim, dim=-1)
   249:         log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
   250:         return TransformedDistribution(
   251:             Normal(mean, torch.exp(log_std)), TanhTransform(cache_size=1)
   252:         ), mean
   253: 
   254:     def forward(self, state: torch.Tensor, deterministic: bool = False):
   255:         dist, mean = self._get_dist(state)
   256:         action = torch.tanh(mean) if deterministic else dist.rsample()
   257:         log_prob = dist.log_prob(action).sum(-1)
   258:         return self.max_action * action, log_prob
   259: 
   260:     def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
   261:         """Log-probability of a dataset action under the current policy."""
   262:         dist, _ = self._get_dist(state)
   263:         action = torch.clamp(action / self.max_action, -1.0 + 1e-6, 1.0 - 1e-6)
   264:         return dist.log_prob(action).sum(-1)
   265: 
   266:     @torch.no_grad()
   267:     def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
   268:         state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
   269:         actions, _ = self(state, not self.training)
   270:         return actions.cpu().data.numpy().flatten()
   271: 
   272: 
   273: class Critic(nn.Module):
   274:     """Q-function Q(s, a). Default: 3 x 256 MLP."""
   275: 
   276:     def __init__(self, state_dim: int, action_dim: int, orthogonal_init: bool = False):
   277:         super().__init__()
   278:         self.net = nn.Sequential(
   279:             nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
   280:             nn.Linear(256, 256), nn.ReLU(),
   281:             nn.Linear(256, 256), nn.ReLU(),
   282:             nn.Linear(256, 1),
   283:         )
   284:         init_module_weights(self.net, orthogonal_init)
   285: 
   286:     def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
   287:         return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)
   288: 
   289: 
   290: class ValueFunction(nn.Module):
   291:     """State value function V(s). Default: 3 x 256 MLP. Useful for IQL-style algorithms."""
   292: 
   293:     def __init__(self, state_dim: int, orthogonal_init: bool = False):
   294:         super().__init__()
   295:         self.net = nn.Sequential(
   296:             nn.Linear(state_dim, 256), nn.ReLU(),
   297:             nn.Linear(256, 256), nn.ReLU(),
   298:             nn.Linear(256, 256), nn.ReLU(),
   299:             nn.Linear(256, 1),
   300:         )
   301:         init_module_weights(self.net, orthogonal_init)
   302: 
   303:     def forward(self, state: torch.Tensor) -> torch.Tensor:
   304:         return self.net(state).squeeze(-1)
   305: 
   306: 
   307: class OfflineAlgorithm:
   308:     """Offline RL algorithm -- implement your approach here.
   309: 
   310:     Goal: suppress Q-value overestimation while learning from offline data
   311:     on continuous locomotion tasks (HalfCheetah, Hopper, Walker2d).
   312: 
   313:     The training loop calls:
   314:         trainer = OfflineAlgorithm(state_dim, action_dim, max_action, **kwargs)
   315:         log_dict = trainer.train(batch)          # called at every timestep
   316:         eval_actor(env, trainer.actor, ...)      # called every eval_freq steps
   317: 
   318:     You MUST set self.actor to an nn.Module that has an .act(state, device) method.
   319:     Build all your networks here; the training loop only provides environment dimensions
   320:     and the hyperparameters from TrainConfig.
   321: 
   322:     actor_lr, critic_lr, alpha_lr are passed from TrainConfig as starting points --
   323:     you may use different values by creating optimizers with your own lr in __init__.
   324:     Any additional algorithm-specific hyperparameters should be hardcoded in __init__.
   325: 
   326:     Dataset access:
   327:         replay_buffer is a ReplayBuffer instance containing the full offline dataset.
   328:         You can use it to compute dataset-level statistics, e.g.:
   329:             replay_buffer._states[:replay_buffer._size]   -- all states  (Tensor)
   330:             replay_buffer._actions[:replay_buffer._size]  -- all actions (Tensor)
   331:             replay_buffer._rewards[:replay_buffer._size]  -- all rewards (Tensor)
   332:             replay_buffer._next_actions[:replay_buffer._size] -- next actions (Tensor)
   333:             replay_buffer._size                           -- number of transitions
   334: 
   335:     Available network classes (defined above, editable):
   336:         DeterministicActor  -- deterministic tanh policy (for BC / TD3+BC approaches)
   337:         Actor               -- stochastic Tanh-Gaussian policy (for CQL / IQL approaches)
   338:         Critic              -- Q(s, a) network
   339:         ValueFunction       -- V(s) network (for IQL-style approaches)
   340:     Available utilities (fixed): soft_update, init_module_weights
   341:     """
   342: 
   343:     def __init__(
   344:         self,
   345:         state_dim: int,
   346:         action_dim: int,
   347:         max_action: float,
   348:         replay_buffer: "ReplayBuffer" = None,
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
   362:         # Full offline dataset -- use for computing global statistics
   363:         # (e.g. action mean/std for normalization, reward scaling, etc.)
   364:         self.replay_buffer = replay_buffer
   365: 
   366:         # Build networks -- modify or replace as needed
   367:         self.actor = Actor(state_dim, action_dim, max_action, orthogonal_init).to(device)
   368:         self.critic_1 = Critic(state_dim, action_dim, orthogonal_init).to(device)
   369:         self.critic_2 = Critic(state_dim, action_dim, orthogonal_init).to(device)
   370:         self.critic_1_target = deepcopy(self.critic_1)
   371:         self.critic_2_target = deepcopy(self.critic_2)
   372: 
   373:         self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
   374:         self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
   375:         self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
   376: 
   377:     def train(self, batch: TensorBatch) -> Dict[str, float]:
   378:         """Update networks on one batch. Return a dict of scalar metrics for logging.
   379: 
   380:         batch = [states, actions, rewards, next_states, dones, next_actions]
   381:         (torch.Tensor, on device)
   382: 
   383:         next_actions is the action at the next timestep in the dataset
   384:         (useful for algorithms like ReBRAC that penalize deviations from
   385:         the data's next action).
   386: 
   387:         TODO: implement your offline RL algorithm here.
   388:         """
   389:         self.total_it += 1
   390:         states, actions, rewards, next_states, dones, next_actions = batch
   391: 
   392:         # -- Placeholder: replace with your algorithm --
   393:         log_dict: Dict[str, float] = {
   394:             "actor_loss": 0.0,
   395:             "critic_loss": 0.0,
   396:         }
   397:         return log_dict
   398: 
   399: 
   400: # =====================================================================
   401: # FIXED: Training loop
   402: # =====================================================================
   403: @pyrallis.wrap()
   404: def train(config: TrainConfig):
   405:     # Apply editable config overrides
   406:     for _k, _v in CONFIG_OVERRIDES.items():
   407:         if hasattr(config, _k):
   408:             setattr(config, _k, _v)
   409: 
   410:     env = gym.make(config.env)
   411:     state_dim = env.observation_space.shape[0]
   412:     action_dim = env.action_space.shape[0]
   413:     max_action = float(env.action_space.high[0])
   414: 
   415:     dataset = d4rl.qlearning_dataset(env)
   416: 
   417:     if config.normalize:
   418:         state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
   419:     else:
   420:         state_mean, state_std = 0.0, 1.0
   421: 
   422:     dataset["observations"] = normalize_states(dataset["observations"], state_mean, state_std)
   423:     dataset["next_observations"] = normalize_states(
   424:         dataset["next_observations"], state_mean, state_std
   425:     )
   426:     env = wrap_env(env, state_mean=state_mean, state_std=state_std)
   427: 
   428:     replay_buffer = ReplayBuffer(state_dim, action_dim, config.buffer_size, config.device)
   429:     replay_buffer.load_d4rl_dataset(dataset)
   430: 
   431:     set_seed(config.seed, env)
   432: 
   433:     trainer = OfflineAlgorithm(
   434:         state_dim=state_dim,
   435:         action_dim=action_dim,
   436:         max_action=max_action,
   437:         replay_buffer=replay_buffer,
   438:         discount=config.discount,
   439:         tau=config.tau,
   440:         actor_lr=config.actor_lr,
   441:         critic_lr=config.critic_lr,
   442:         alpha_lr=config.alpha_lr,
   443:         orthogonal_init=config.orthogonal_init,
   444:         device=config.device,
   445:     )
   446: 
   447:     if config.checkpoints_path is not None:
   448:         print(f"Checkpoints path: {config.checkpoints_path}")
   449:         os.makedirs(config.checkpoints_path, exist_ok=True)
   450: 
   451:     evaluations = []
   452:     for t in range(int(config.max_timesteps)):
   453:         batch = replay_buffer.sample(config.batch_size)
   454:         batch = [b.to(config.device) for b in batch]
   455:         log_dict = trainer.train(batch)
   456: 
   457:         if (t + 1) % 1000 == 0:
   458:             metrics_str = " ".join(
   459:                 f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
   460:                 for k, v in log_dict.items()
   461:             )
   462:             print(f"TRAIN_METRICS step={t+1} {metrics_str}", flush=True)
   463: 
   464:         if (t + 1) % config.eval_freq == 0:
   465:             print(f"Time steps: {t + 1}")
   466:             eval_scores = eval_actor(
   467:                 env, trainer.actor, device=config.device,
   468:                 n_episodes=config.n_episodes, seed=config.seed,
   469:             )
   470:             eval_score = eval_scores.mean()
   471:             normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
   472:             evaluations.append(normalized_eval_score)
   473:             print("---------------------------------------")
   474:             print(
   475:                 f"Evaluation over {config.n_episodes} episodes: "
   476:                 f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}"
   477:             )
   478:             print("---------------------------------------")
   479:             if config.checkpoints_path is not None:
   480:                 pass  # checkpoint saving disabled (crashes on tmpfs)
   481: 
   482: 
   483: if __name__ == "__main__":
   484:     train()
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


### `rebrac` baseline — editable region  [READ-ONLY — reference implementation]

In `CORL/algorithms/offline/custom.py`:

```python
Lines 193–435:
   190: # EDITABLE: Network definitions and OfflineAlgorithm
   191: #
   192: # CONSTRAINTS:
   193: # - Total trainable parameter count is soft-capped.
   194: # - Total parameter count is checked at runtime and must not exceed
   195: #   1.2x the largest baseline. Focus on algorithmic improvements, not
   196: #   network capacity.
   197: #
   198: # CONFIG_OVERRIDES: set any TrainConfig field here to override the fixed
   199: # defaults. Allowed keys: normalize, normalize_reward, actor_lr, critic_lr,
   200: # tau, discount, batch_size.
   201: # =====================================================================
   202: import sys as _sys
   203: 
   204: def _detect_env():
   205:     """Parse --env from sys.argv to determine environment name."""
   206:     for i, arg in enumerate(_sys.argv):
   207:         if arg == "--env" and i + 1 < len(_sys.argv):
   208:             return _sys.argv[i + 1]
   209:         if arg.startswith("--env="):
   210:             return arg.split("=", 1)[1]
   211:     return ""
   212: 
   213: _REBRAC_ENV = _detect_env()
   214: 
   215: # Per-environment ReBRAC hyperparameters for this benchmark harness
   216: _REBRAC_HPARAMS = {
   217:     "halfcheetah-medium-v2": {"actor_bc_coef": 0.001, "critic_bc_coef": 0.01,  "lr": 1e-3, "batch_size": 1024},
   218:     "walker2d-medium-v2":    {"actor_bc_coef": 0.05,  "critic_bc_coef": 0.1,   "lr": 1e-3, "batch_size": 1024},
   219:     "hopper-medium-v2":      {"actor_bc_coef": 0.01,  "critic_bc_coef": 0.01,  "lr": 1e-3, "batch_size": 1024},
   220:     "maze2d-large-v1":       {"actor_bc_coef": 0.003, "critic_bc_coef": 0.001, "lr": 3e-4, "batch_size": 256},
   221:     "maze2d-medium-v1":      {"actor_bc_coef": 0.003, "critic_bc_coef": 0.001, "lr": 3e-4, "batch_size": 256},
   222: }
   223: _REBRAC_HP = _REBRAC_HPARAMS.get(_REBRAC_ENV, {"actor_bc_coef": 0.01, "critic_bc_coef": 0.01, "lr": 1e-3, "batch_size": 1024})
   224: 
   225: CONFIG_OVERRIDES: Dict[str, Any] = {}
   226: class DeterministicActor(nn.Module):
   227:     """Deterministic policy pi(s) = tanh(net(s)) * max_action.
   228:     ReBRAC-style: 3 x 256 MLP without LayerNorm (matching CORL actor_ln=False)."""
   229: 
   230:     def __init__(self, state_dim: int, action_dim: int, max_action: float):
   231:         super().__init__()
   232:         self.max_action = max_action
   233:         self.net = nn.Sequential(
   234:             nn.Linear(state_dim, 256), nn.ReLU(),
   235:             nn.Linear(256, 256), nn.ReLU(),
   236:             nn.Linear(256, 256), nn.ReLU(),
   237:             nn.Linear(256, action_dim), nn.Tanh(),
   238:         )
   239: 
   240:     def forward(self, state: torch.Tensor) -> torch.Tensor:
   241:         return self.max_action * self.net(state)
   242: 
   243:     @torch.no_grad()
   244:     def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
   245:         state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
   246:         return self(state).cpu().data.numpy().flatten()
   247: 
   248: 
   249: class Actor(nn.Module):
   250:     """Tanh-Gaussian stochastic policy. Default: 3 x 256 MLP.
   251:     Suitable for CQL, IQL style algorithms."""
   252: 
   253:     def __init__(self, state_dim: int, action_dim: int, max_action: float,
   254:                  orthogonal_init: bool = False):
   255:         super().__init__()
   256:         self.max_action = max_action
   257:         self.action_dim = action_dim
   258:         self.net = nn.Sequential(
   259:             nn.Linear(state_dim, 256), nn.ReLU(),
   260:             nn.Linear(256, 256), nn.ReLU(),
   261:             nn.Linear(256, 256), nn.ReLU(),
   262:             nn.Linear(256, 2 * action_dim),
   263:         )
   264:         init_module_weights(self.net, orthogonal_init)
   265:         self.log_std_min = -20.0
   266:         self.log_std_max = 2.0
   267: 
   268:     def _get_dist(self, state: torch.Tensor):
   269:         out = self.net(state)
   270:         mean, log_std = torch.split(out, self.action_dim, dim=-1)
   271:         log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
   272:         return TransformedDistribution(
   273:             Normal(mean, torch.exp(log_std)), TanhTransform(cache_size=1)
   274:         ), mean
   275: 
   276:     def forward(self, state: torch.Tensor, deterministic: bool = False):
   277:         dist, mean = self._get_dist(state)
   278:         action = torch.tanh(mean) if deterministic else dist.rsample()
   279:         log_prob = dist.log_prob(action).sum(-1)
   280:         return self.max_action * action, log_prob
   281: 
   282:     def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
   283:         """Log-probability of a dataset action under the current policy."""
   284:         dist, _ = self._get_dist(state)
   285:         action = torch.clamp(action / self.max_action, -1.0 + 1e-6, 1.0 - 1e-6)
   286:         return dist.log_prob(action).sum(-1)
   287: 
   288:     @torch.no_grad()
   289:     def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
   290:         state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
   291:         actions, _ = self(state, not self.training)
   292:         return actions.cpu().data.numpy().flatten()
   293: 
   294: class Critic(nn.Module):
   295:     """Q-function Q(s, a). 3 x 256 MLP with LayerNorm (ReBRAC-style, critic_ln=True)."""
   296: 
   297:     def __init__(self, state_dim: int, action_dim: int, orthogonal_init: bool = False):
   298:         super().__init__()
   299:         self.net = nn.Sequential(
   300:             nn.Linear(state_dim + action_dim, 256), nn.ReLU(), nn.LayerNorm(256),
   301:             nn.Linear(256, 256), nn.ReLU(), nn.LayerNorm(256),
   302:             nn.Linear(256, 256), nn.ReLU(), nn.LayerNorm(256),
   303:             nn.Linear(256, 1),
   304:         )
   305: 
   306:     def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
   307:         return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)
   308: 
   309: 
   310: class ValueFunction(nn.Module):
   311:     """State value function V(s). Default: 3 x 256 MLP. Useful for IQL-style algorithms."""
   312: 
   313:     def __init__(self, state_dim: int, orthogonal_init: bool = False):
   314:         super().__init__()
   315:         self.net = nn.Sequential(
   316:             nn.Linear(state_dim, 256), nn.ReLU(),
   317:             nn.Linear(256, 256), nn.ReLU(),
   318:             nn.Linear(256, 256), nn.ReLU(),
   319:             nn.Linear(256, 1),
   320:         )
   321:         init_module_weights(self.net, orthogonal_init)
   322: 
   323:     def forward(self, state: torch.Tensor) -> torch.Tensor:
   324:         return self.net(state).squeeze(-1)
   325: 
   326: class OfflineAlgorithm:
   327:     """ReBRAC — Regularized Behavior Regularized Actor Critic.
   328: 
   329:     TD3-style with BC penalties on actor and critic targets.
   330:     Per-environment BC coefficients and learning rates from CORL configs.
   331:     """
   332: 
   333:     def __init__(
   334:         self,
   335:         state_dim: int,
   336:         action_dim: int,
   337:         max_action: float,
   338:         replay_buffer=None,
   339:         discount: float = 0.99,
   340:         tau: float = 5e-3,
   341:         actor_lr: float = 3e-4,
   342:         critic_lr: float = 3e-4,
   343:         alpha_lr: float = 3e-4,
   344:         orthogonal_init: bool = True,
   345:         device: str = "cuda",
   346:     ):
   347:         self.device = device
   348:         self.discount = discount
   349:         self.tau = tau
   350:         self.max_action = max_action
   351:         self.total_it = 0
   352: 
   353:         # Per-env tuned ReBRAC hyperparameters for this benchmark harness
   354:         self.actor_bc_coef = _REBRAC_HP["actor_bc_coef"]
   355:         self.critic_bc_coef = _REBRAC_HP["critic_bc_coef"]
   356:         _lr = _REBRAC_HP["lr"]
   357:         self.policy_noise = 0.2      # target policy smoothing noise
   358:         self.noise_clip = 0.5        # clipping range for smoothing noise
   359:         self.policy_freq = 2         # delayed actor update frequency
   360:         self.normalize_q = True      # normalize Q in actor loss
   361: 
   362:         # Actor (deterministic, no LayerNorm) + target
   363:         self.actor = DeterministicActor(state_dim, action_dim, max_action).to(device)
   364:         self.actor_target = deepcopy(self.actor)
   365:         self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=_lr)
   366: 
   367:         # Twin critics (with LayerNorm) + targets
   368:         self.critic_1 = Critic(state_dim, action_dim, orthogonal_init).to(device)
   369:         self.critic_1_target = deepcopy(self.critic_1)
   370:         self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=_lr)
   371: 
   372:         self.critic_2 = Critic(state_dim, action_dim, orthogonal_init).to(device)
   373:         self.critic_2_target = deepcopy(self.critic_2)
   374:         self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=_lr)
   375: 
   376:     def train(self, batch: TensorBatch) -> Dict[str, float]:
   377:         self.total_it += 1
   378:         states, actions, rewards, next_states, dones, next_actions_data = batch
   379:         not_done = 1 - dones.squeeze(-1)
   380:         rewards_flat = rewards.squeeze(-1)
   381:         log_dict: Dict[str, float] = {}
   382: 
   383:         # ── Critic update ──────────────────────────────────────────────
   384:         with torch.no_grad():
   385:             noise = (torch.randn_like(actions) * self.policy_noise).clamp(
   386:                 -self.noise_clip, self.noise_clip
   387:             )
   388:             next_actions = (self.actor_target(next_states) + noise).clamp(
   389:                 -self.max_action, self.max_action
   390:             )
   391:             # BC penalty on next actions (compare policy's next actions to dataset's)
   392:             bc_penalty = ((next_actions - next_actions_data) ** 2).sum(-1)
   393: 
   394:             target_q1 = self.critic_1_target(next_states, next_actions)
   395:             target_q2 = self.critic_2_target(next_states, next_actions)
   396:             target_q = torch.min(target_q1, target_q2)
   397:             # Subtract BC penalty from critic target (ReBRAC key idea)
   398:             target_q = target_q - self.critic_bc_coef * bc_penalty
   399:             target_q = rewards_flat + not_done * self.discount * target_q
   400: 
   401:         current_q1 = self.critic_1(states, actions)
   402:         current_q2 = self.critic_2(states, actions)
   403:         critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
   404:         log_dict["critic_loss"] = critic_loss.item()
   405: 
   406:         self.critic_1_optimizer.zero_grad()
   407:         self.critic_2_optimizer.zero_grad()
   408:         critic_loss.backward()
   409:         self.critic_1_optimizer.step()
   410:         self.critic_2_optimizer.step()
   411: 
   412:         # ── Delayed actor update ───────────────────────────────────────
   413:         if self.total_it % self.policy_freq == 0:
   414:             pi = self.actor(states)
   415:             q = self.critic_1(states, pi)
   416: 
   417:             # BC penalty on actor
   418:             bc_mse = ((pi - actions) ** 2).sum(-1)
   419: 
   420:             lmbda = 1.0
   421:             if self.normalize_q:
   422:                 lmbda = 1.0 / (torch.abs(q).mean().detach() + 1e-8)
   423: 
   424:             actor_loss = (self.actor_bc_coef * bc_mse - lmbda * q).mean()
   425:             log_dict["actor_loss"] = actor_loss.item()
   426: 
   427:             self.actor_optimizer.zero_grad()
   428:             actor_loss.backward()
   429:             self.actor_optimizer.step()
   430: 
   431:             soft_update(self.critic_1_target, self.critic_1, self.tau)
   432:             soft_update(self.critic_2_target, self.critic_2, self.tau)
   433:             soft_update(self.actor_target, self.actor, self.tau)
   434: 
   435:         return log_dict
   436: 
   437: 
   438: # =====================================================================
```

### `td3_bc` baseline — editable region  [READ-ONLY — reference implementation]

In `CORL/algorithms/offline/custom.py`:

```python
Lines 193–393:
   190: # EDITABLE: Network definitions and OfflineAlgorithm
   191: #
   192: # CONSTRAINTS:
   193: # - Total trainable parameter count is soft-capped.
   194: # - Total parameter count is checked at runtime and must not exceed
   195: #   1.2x the largest baseline. Focus on algorithmic improvements, not
   196: #   network capacity.
   197: #
   198: # CONFIG_OVERRIDES: override method-specific TrainConfig fields here.
   199: # Allowed keys: normalize, normalize_reward, actor_lr, critic_lr, tau, discount.
   200: # Example: CONFIG_OVERRIDES = {"normalize": False, "actor_lr": 1e-3}
   201: # =====================================================================
   202: CONFIG_OVERRIDES: Dict[str, Any] = {}
   203: 
   204: 
   205: class DeterministicActor(nn.Module):
   206:     """Deterministic policy pi(s) = tanh(net(s)) * max_action.
   207:     Suitable for BC, TD3+BC style algorithms. Default: 2 x 256 MLP."""
   208: 
   209:     def __init__(self, state_dim: int, action_dim: int, max_action: float):
   210:         super().__init__()
   211:         self.max_action = max_action
   212:         self.net = nn.Sequential(
   213:             nn.Linear(state_dim, 256), nn.ReLU(),
   214:             nn.Linear(256, 256), nn.ReLU(),
   215:             nn.Linear(256, action_dim), nn.Tanh(),
   216:         )
   217: 
   218:     def forward(self, state: torch.Tensor) -> torch.Tensor:
   219:         return self.max_action * self.net(state)
   220: 
   221:     @torch.no_grad()
   222:     def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
   223:         state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
   224:         return self(state).cpu().data.numpy().flatten()
   225: 
   226: 
   227: class Actor(nn.Module):
   228:     """Tanh-Gaussian stochastic policy. Default: 3 x 256 MLP.
   229:     Suitable for CQL, IQL style algorithms."""
   230: 
   231:     def __init__(self, state_dim: int, action_dim: int, max_action: float,
   232:                  orthogonal_init: bool = False):
   233:         super().__init__()
   234:         self.max_action = max_action
   235:         self.action_dim = action_dim
   236:         self.net = nn.Sequential(
   237:             nn.Linear(state_dim, 256), nn.ReLU(),
   238:             nn.Linear(256, 256), nn.ReLU(),
   239:             nn.Linear(256, 256), nn.ReLU(),
   240:             nn.Linear(256, 2 * action_dim),
   241:         )
   242:         init_module_weights(self.net, orthogonal_init)
   243:         self.log_std_min = -20.0
   244:         self.log_std_max = 2.0
   245: 
   246:     def _get_dist(self, state: torch.Tensor):
   247:         out = self.net(state)
   248:         mean, log_std = torch.split(out, self.action_dim, dim=-1)
   249:         log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
   250:         return TransformedDistribution(
   251:             Normal(mean, torch.exp(log_std)), TanhTransform(cache_size=1)
   252:         ), mean
   253: 
   254:     def forward(self, state: torch.Tensor, deterministic: bool = False):
   255:         dist, mean = self._get_dist(state)
   256:         action = torch.tanh(mean) if deterministic else dist.rsample()
   257:         log_prob = dist.log_prob(action).sum(-1)
   258:         return self.max_action * action, log_prob
   259: 
   260:     def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
   261:         """Log-probability of a dataset action under the current policy."""
   262:         dist, _ = self._get_dist(state)
   263:         action = torch.clamp(action / self.max_action, -1.0 + 1e-6, 1.0 - 1e-6)
   264:         return dist.log_prob(action).sum(-1)
   265: 
   266:     @torch.no_grad()
   267:     def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
   268:         state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
   269:         actions, _ = self(state, not self.training)
   270:         return actions.cpu().data.numpy().flatten()
   271: 
   272: class Critic(nn.Module):
   273:     """Q-function Q(s, a). 2 × 256 MLP (TD3+BC reference architecture)."""
   274: 
   275:     def __init__(self, state_dim: int, action_dim: int, orthogonal_init: bool = False):
   276:         super().__init__()
   277:         self.net = nn.Sequential(
   278:             nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
   279:             nn.Linear(256, 256), nn.ReLU(),
   280:             nn.Linear(256, 1),
   281:         )
   282: 
   283:     def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
   284:         return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)
   285: 
   286: 
   287: class ValueFunction(nn.Module):
   288:     """State value function V(s). Default: 3 x 256 MLP. Useful for IQL-style algorithms."""
   289: 
   290:     def __init__(self, state_dim: int, orthogonal_init: bool = False):
   291:         super().__init__()
   292:         self.net = nn.Sequential(
   293:             nn.Linear(state_dim, 256), nn.ReLU(),
   294:             nn.Linear(256, 256), nn.ReLU(),
   295:             nn.Linear(256, 256), nn.ReLU(),
   296:             nn.Linear(256, 1),
   297:         )
   298:         init_module_weights(self.net, orthogonal_init)
   299: 
   300:     def forward(self, state: torch.Tensor) -> torch.Tensor:
   301:         return self.net(state).squeeze(-1)
   302: 
   303: class OfflineAlgorithm:
   304:     """TD3+BC — Twin Delayed DDPG with Behavior Cloning regularization."""
   305: 
   306:     def __init__(
   307:         self,
   308:         state_dim: int,
   309:         action_dim: int,
   310:         max_action: float,
   311:         replay_buffer=None,
   312:         discount: float = 0.99,
   313:         tau: float = 5e-3,
   314:         actor_lr: float = 3e-4,
   315:         critic_lr: float = 3e-4,
   316:         alpha_lr: float = 3e-4,
   317:         orthogonal_init: bool = True,
   318:         device: str = "cuda",
   319:     ):
   320:         self.device = device
   321:         self.discount = discount
   322:         self.tau = tau
   323:         self.max_action = max_action
   324:         self.total_it = 0
   325: 
   326:         # TD3+BC hyperparameters
   327:         self.alpha = 2.5
   328:         self.policy_noise = 0.2 * max_action
   329:         self.noise_clip = 0.5 * max_action
   330:         self.policy_freq = 2
   331: 
   332:         # Actor (deterministic) + target
   333:         self.actor = DeterministicActor(state_dim, action_dim, max_action).to(device)
   334:         self.actor_target = deepcopy(self.actor)
   335:         self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
   336: 
   337:         # Twin critics + targets
   338:         self.critic_1 = Critic(state_dim, action_dim, orthogonal_init).to(device)
   339:         self.critic_1_target = deepcopy(self.critic_1)
   340:         self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
   341: 
   342:         self.critic_2 = Critic(state_dim, action_dim, orthogonal_init).to(device)
   343:         self.critic_2_target = deepcopy(self.critic_2)
   344:         self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
   345: 
   346:     def train(self, batch: TensorBatch) -> Dict[str, float]:
   347:         self.total_it += 1
   348:         states, actions, rewards, next_states, dones, *_ = batch
   349:         not_done = 1 - dones.squeeze(-1)
   350:         rewards_flat = rewards.squeeze(-1)
   351:         log_dict: Dict[str, float] = {}
   352: 
   353:         with torch.no_grad():
   354:             noise = (torch.randn_like(actions) * self.policy_noise).clamp(
   355:                 -self.noise_clip, self.noise_clip
   356:             )
   357:             next_action = (self.actor_target(next_states) + noise).clamp(
   358:                 -self.max_action, self.max_action
   359:             )
   360:             target_q1 = self.critic_1_target(next_states, next_action)
   361:             target_q2 = self.critic_2_target(next_states, next_action)
   362:             target_q = torch.min(target_q1, target_q2)
   363:             target_q = rewards_flat + not_done * self.discount * target_q
   364: 
   365:         current_q1 = self.critic_1(states, actions)
   366:         current_q2 = self.critic_2(states, actions)
   367:         critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
   368:         log_dict["critic_loss"] = critic_loss.item()
   369: 
   370:         self.critic_1_optimizer.zero_grad()
   371:         self.critic_2_optimizer.zero_grad()
   372:         critic_loss.backward()
   373:         self.critic_1_optimizer.step()
   374:         self.critic_2_optimizer.step()
   375: 
   376:         # Delayed actor updates
   377:         if self.total_it % self.policy_freq == 0:
   378:             pi = self.actor(states)
   379:             q = self.critic_1(states, pi)
   380:             lmbda = self.alpha / q.abs().mean().detach()
   381: 
   382:             actor_loss = -lmbda * q.mean() + F.mse_loss(pi, actions)
   383:             log_dict["actor_loss"] = actor_loss.item()
   384: 
   385:             self.actor_optimizer.zero_grad()
   386:             actor_loss.backward()
   387:             self.actor_optimizer.step()
   388: 
   389:             soft_update(self.critic_1_target, self.critic_1, self.tau)
   390:             soft_update(self.critic_2_target, self.critic_2, self.tau)
   391:             soft_update(self.actor_target, self.actor, self.tau)
   392: 
   393:         return log_dict
   394: 
   395: 
   396: # =====================================================================
```

### `iql` baseline — editable region  [READ-ONLY — reference implementation]

In `CORL/algorithms/offline/custom.py`:

```python
Lines 193–395:
   190: # EDITABLE: Network definitions and OfflineAlgorithm
   191: #
   192: # CONSTRAINTS:
   193: # - Total trainable parameter count is soft-capped.
   194: # - Total parameter count is checked at runtime and must not exceed
   195: #   1.2x the largest baseline. Focus on algorithmic improvements, not
   196: #   network capacity.
   197: #
   198: # CONFIG_OVERRIDES: override method-specific TrainConfig fields here.
   199: # Allowed keys: normalize, normalize_reward, actor_lr, critic_lr, tau, discount.
   200: # Example: CONFIG_OVERRIDES = {"normalize": False, "actor_lr": 1e-3}
   201: # =====================================================================
   202: CONFIG_OVERRIDES: Dict[str, Any] = {}
   203: 
   204: from torch.optim.lr_scheduler import CosineAnnealingLR
   205: 
   206: EXP_ADV_MAX = 100.0
   207: 
   208: def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
   209:     return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)
   210: 
   211: 
   212: class DeterministicActor(nn.Module):
   213:     """Deterministic policy pi(s) = tanh(net(s)) * max_action.
   214:     Suitable for BC, TD3+BC style algorithms. Default: 2 x 256 MLP."""
   215: 
   216:     def __init__(self, state_dim: int, action_dim: int, max_action: float):
   217:         super().__init__()
   218:         self.max_action = max_action
   219:         self.net = nn.Sequential(
   220:             nn.Linear(state_dim, 256), nn.ReLU(),
   221:             nn.Linear(256, 256), nn.ReLU(),
   222:             nn.Linear(256, action_dim), nn.Tanh(),
   223:         )
   224: 
   225:     def forward(self, state: torch.Tensor) -> torch.Tensor:
   226:         return self.max_action * self.net(state)
   227: 
   228:     @torch.no_grad()
   229:     def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
   230:         state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
   231:         return self(state).cpu().data.numpy().flatten()
   232: 
   233: class Actor(nn.Module):
   234:     """GaussianPolicy for IQL — state-independent log_std, forward returns Normal."""
   235: 
   236:     def __init__(self, state_dim: int, action_dim: int, max_action: float,
   237:                  orthogonal_init: bool = False):
   238:         super().__init__()
   239:         self.max_action = max_action
   240:         self.action_dim = action_dim
   241:         # 2-hidden-layer MLP with Tanh output (matching IQL reference)
   242:         self.net = nn.Sequential(
   243:             nn.Linear(state_dim, 256), nn.ReLU(),
   244:             nn.Linear(256, 256), nn.ReLU(),
   245:             nn.Linear(256, action_dim), nn.Tanh(),
   246:         )
   247:         self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
   248:         self.log_std_min = -20.0
   249:         self.log_std_max = 2.0
   250: 
   251:     def forward(self, state: torch.Tensor) -> Normal:
   252:         mean = self.net(state)
   253:         std = torch.exp(self.log_std.clamp(self.log_std_min, self.log_std_max))
   254:         return Normal(mean, std)
   255: 
   256:     @torch.no_grad()
   257:     def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
   258:         state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
   259:         dist = self(state)
   260:         action = dist.mean if not self.training else dist.sample()
   261:         action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
   262:         return action.cpu().data.numpy().flatten()
   263: 
   264: class Critic(nn.Module):
   265:     """Q-function Q(s, a). 2 × 256 MLP (IQL reference architecture)."""
   266: 
   267:     def __init__(self, state_dim: int, action_dim: int, orthogonal_init: bool = False):
   268:         super().__init__()
   269:         self.net = nn.Sequential(
   270:             nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
   271:             nn.Linear(256, 256), nn.ReLU(),
   272:             nn.Linear(256, 1),
   273:         )
   274: 
   275:     def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
   276:         return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)
   277: 
   278: class ValueFunction(nn.Module):
   279:     """State value function V(s). 2 × 256 MLP (IQL reference architecture)."""
   280: 
   281:     def __init__(self, state_dim: int, orthogonal_init: bool = False):
   282:         super().__init__()
   283:         self.net = nn.Sequential(
   284:             nn.Linear(state_dim, 256), nn.ReLU(),
   285:             nn.Linear(256, 256), nn.ReLU(),
   286:             nn.Linear(256, 1),
   287:         )
   288: 
   289:     def forward(self, state: torch.Tensor) -> torch.Tensor:
   290:         return self.net(state).squeeze(-1)
   291: 
   292: class OfflineAlgorithm:
   293:     """IQL — Implicit Q-Learning with expectile regression and advantage-weighted actor."""
   294: 
   295:     def __init__(
   296:         self,
   297:         state_dim: int,
   298:         action_dim: int,
   299:         max_action: float,
   300:         replay_buffer=None,
   301:         discount: float = 0.99,
   302:         tau: float = 5e-3,
   303:         actor_lr: float = 3e-4,
   304:         critic_lr: float = 3e-4,
   305:         alpha_lr: float = 3e-4,
   306:         orthogonal_init: bool = True,
   307:         device: str = "cuda",
   308:     ):
   309:         self.device = device
   310:         self.discount = discount
   311:         self.tau = tau
   312:         self.max_action = max_action
   313:         self.total_it = 0
   314: 
   315:         # IQL hyperparameters
   316:         self.beta = 3.0
   317:         self.iql_tau = 0.7
   318: 
   319:         # Actor (GaussianPolicy-style via replaced Actor class)
   320:         self.actor = Actor(state_dim, action_dim, max_action).to(device)
   321:         self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
   322:         self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, int(1e6))
   323: 
   324:         # Twin Q via two separate Critic instances + targets
   325:         self.critic_1 = Critic(state_dim, action_dim, orthogonal_init).to(device)
   326:         self.critic_2 = Critic(state_dim, action_dim, orthogonal_init).to(device)
   327:         self.critic_1_target = deepcopy(self.critic_1).requires_grad_(False).to(device)
   328:         self.critic_2_target = deepcopy(self.critic_2).requires_grad_(False).to(device)
   329:         self.q_optimizer = torch.optim.Adam(
   330:             list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
   331:             lr=critic_lr,
   332:         )
   333: 
   334:         # Value function V(s)
   335:         self.vf = ValueFunction(state_dim, orthogonal_init).to(device)
   336:         self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=critic_lr)
   337: 
   338:     def _update_v(self, observations, actions, log_dict):
   339:         with torch.no_grad():
   340:             target_q = torch.min(
   341:                 self.critic_1_target(observations, actions),
   342:                 self.critic_2_target(observations, actions),
   343:             )
   344:         v = self.vf(observations)
   345:         adv = target_q - v
   346:         v_loss = asymmetric_l2_loss(adv, self.iql_tau)
   347:         log_dict["value_loss"] = v_loss.item()
   348:         self.v_optimizer.zero_grad()
   349:         v_loss.backward()
   350:         self.v_optimizer.step()
   351:         return adv
   352: 
   353:     def _update_q(self, next_v, observations, actions, rewards, dones, log_dict):
   354:         targets = rewards + (1.0 - dones.float()) * self.discount * next_v.detach()
   355:         q1 = self.critic_1(observations, actions)
   356:         q2 = self.critic_2(observations, actions)
   357:         q_loss = (F.mse_loss(q1, targets) + F.mse_loss(q2, targets)) / 2.0
   358:         log_dict["q_loss"] = q_loss.item()
   359:         self.q_optimizer.zero_grad()
   360:         q_loss.backward()
   361:         self.q_optimizer.step()
   362:         soft_update(self.critic_1_target, self.critic_1, self.tau)
   363:         soft_update(self.critic_2_target, self.critic_2, self.tau)
   364: 
   365:     def _update_policy(self, adv, observations, actions, log_dict):
   366:         exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
   367:         policy_out = self.actor(observations)
   368:         if isinstance(policy_out, torch.distributions.Distribution):
   369:             bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
   370:         elif torch.is_tensor(policy_out):
   371:             bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
   372:         else:
   373:             raise NotImplementedError
   374:         policy_loss = torch.mean(exp_adv * bc_losses)
   375:         log_dict["actor_loss"] = policy_loss.item()
   376:         self.actor_optimizer.zero_grad()
   377:         policy_loss.backward()
   378:         self.actor_optimizer.step()
   379:         self.actor_lr_schedule.step()
   380: 
   381:     def train(self, batch: TensorBatch) -> Dict[str, float]:
   382:         self.total_it += 1
   383:         observations, actions, rewards, next_observations, dones, *_ = batch
   384:         log_dict: Dict[str, float] = {}
   385: 
   386:         with torch.no_grad():
   387:             next_v = self.vf(next_observations)
   388: 
   389:         adv = self._update_v(observations, actions, log_dict)
   390:         rewards = rewards.squeeze(dim=-1)
   391:         dones = dones.squeeze(dim=-1)
   392:         self._update_q(next_v, observations, actions, rewards, dones, log_dict)
   393:         self._update_policy(adv, observations, actions, log_dict)
   394: 
   395:         return log_dict
   396: 
   397: 
   398: # =====================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
