# MLS-Bench: rl-intrinsic-exploration

# RL Intrinsic Exploration: Sparse-Reward Novelty Bonus Design

## Research Question
Design an intrinsic exploration mechanism that improves sparse-reward
discovery in hard-exploration Atari environments.

## Background
In sparse-reward reinforcement learning, external rewards arrive too
infrequently for vanilla policy optimization to learn efficiently. A
common solution is to add an **intrinsic reward** that encourages
novelty, surprise, or state-space coverage on top of the (possibly
clipped) extrinsic reward.

This task isolates that question. The PPO training loop, Atari
preprocessing (grayscale, frame-skip, frame-stack, terminal-on-life-loss),
policy/value architecture, and optimizer are fixed. The only thing you
should redesign is the intrinsic-bonus module and how its signal is mixed
into learning.

Reference families include:
- **No bonus / vanilla PPO** — Schulman et al., "Proximal Policy
  Optimization Algorithms" (arXiv:1707.06347). Learns only from clipped
  extrinsic reward.
- **RND** — Burda et al., "Exploration by Random Network Distillation"
  (arXiv:1810.12894, ICLR 2019). Bonus is the prediction error of a
  learned network against a fixed randomly-initialized target network.
- **ICM** — Pathak et al., "Curiosity-driven Exploration by
  Self-supervised Prediction" (arXiv:1705.05363, ICML 2017). Bonus is the
  forward-dynamics prediction error in a feature space learned by an
  inverse-dynamics model.

## Editable Interface
You will modify the editable section of `custom_intrinsic_exploration.py`:
- `IntrinsicBonusModule` — defines how intrinsic rewards are computed and
  trained.
- `mix_advantages(...)` — defines how extrinsic and intrinsic advantages
  are combined.

The editable code must keep the public interface intact:
- `initialize(envs)`
- `trainable_parameters()`
- `update_batch_stats(batch_obs, batch_next_obs)`
- `compute_bonus(obs, next_obs, actions)`
- `normalize_rollout_rewards(rollout_intrinsic)`
- `loss(batch_obs, batch_next_obs, batch_actions)`
- `mix_advantages(ext_advantages, int_advantages, args)`

## Evaluation
The agent is trained with the same fixed PPO-style loop on multiple
sparse-reward Atari environments, including:
- **Tutankham-v5** — medium-difficulty visible game.
- **Frostbite-v5** — hard-exploration visible game.
- **PrivateEye-v5** — additional hard-exploration setting.

Reported metrics:
- `eval_return` — mean evaluation episodic return at the fixed training
  budget.
- `auc` — area under the evaluation-return curve across training.
- `nonzero_rate` — fraction of evaluation episodes with non-zero
  episodic return.

Evaluation uses deterministic rollouts with a fixed per-episode step
cap so that non-terminating Atari behavior cannot stall the benchmark.
Higher is better for all metrics. A method should improve across
multiple games rather than helping only one.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/cleanrl/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to score zero.

- `cleanrl/cleanrl/custom_intrinsic_exploration.py`
- editable lines **179–219**


Other files you may **read** for context (do not modify):
- `cleanrl/cleanrl/ppo_rnd_envpool.py`
- `cleanrl/cleanrl/ppo_atari_envpool.py`


## Readable Context


### `cleanrl/cleanrl/custom_intrinsic_exploration.py`  [EDITABLE — lines 179–219 only]

```python
     1: # Custom sparse-reward Atari exploration benchmark for MLS-Bench.
     2: #
     3: # FIXED sections: PPO loop, Atari preprocessing, policy/value architecture,
     4: # evaluation, logging, and optimizer wiring.
     5: # EDITABLE section: IntrinsicBonusModule + mix_advantages.
     6: 
     7: from __future__ import annotations
     8: 
     9: import os
    10: import random
    11: import time
    12: from collections import deque
    13: from dataclasses import dataclass
    14: 
    15: import envpool
    16: import gym
    17: import numpy as np
    18: import torch
    19: import torch.nn as nn
    20: import torch.nn.functional as F
    21: import torch.optim as optim
    22: import tyro
    23: from gym.wrappers.normalize import RunningMeanStd
    24: from torch.distributions.categorical import Categorical
    25: 
    26: 
    27: @dataclass
    28: class Args:
    29:     exp_name: str = os.path.basename(__file__)[: -len(".py")]
    30:     seed: int = 1
    31:     torch_deterministic: bool = True
    32:     cuda: bool = True
    33: 
    34:     env_id: str = "MontezumaRevenge-v5"
    35:     total_timesteps: int = 10000000
    36:     learning_rate: float = 1e-4
    37:     num_envs: int = 32
    38:     num_steps: int = 128
    39:     anneal_lr: bool = True
    40:     gamma: float = 0.999
    41:     gae_lambda: float = 0.95
    42:     num_minibatches: int = 4
    43:     update_epochs: int = 4
    44:     norm_adv: bool = True
    45:     clip_coef: float = 0.1
    46:     clip_vloss: bool = True
    47:     ent_coef: float = 0.001
    48:     vf_coef: float = 0.5
    49:     max_grad_norm: float = 0.5
    50:     target_kl: float | None = None
    51: 
    52:     int_coef: float = 1.0
    53:     ext_coef: float = 2.0
    54:     int_gamma: float = 0.99
    55:     update_proportion: float = 0.25
    56:     num_iterations_obs_norm_init: int = 10
    57: 
    58:     eval_interval: int = 500000
    59:     eval_episodes: int = 5
    60:     eval_max_episode_steps: int = 27000
    61: 
    62:     batch_size: int = 0
    63:     minibatch_size: int = 0
    64:     num_iterations: int = 0
    65: 
    66: 
    67: class RecordEpisodeStatistics(gym.Wrapper):
    68:     def __init__(self, env):
    69:         super().__init__(env)
    70:         self.num_envs = getattr(env, "num_envs", 1)
    71:         self.episode_returns = None
    72:         self.episode_lengths = None
    73: 
    74:     def reset(self, **kwargs):
    75:         observations = super().reset(**kwargs)
    76:         self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
    77:         self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
    78:         self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
    79:         self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
    80:         return observations
    81: 
    82:     def step(self, action):
    83:         observations, rewards, dones, infos = super().step(action)
    84:         self.episode_returns += infos["reward"]
    85:         self.episode_lengths += 1
    86:         self.returned_episode_returns[:] = self.episode_returns
    87:         self.returned_episode_lengths[:] = self.episode_lengths
    88:         self.episode_returns *= 1 - infos["terminated"]
    89:         self.episode_lengths *= 1 - infos["terminated"]
    90:         infos["r"] = self.returned_episode_returns
    91:         infos["l"] = self.returned_episode_lengths
    92:         return observations, rewards, dones, infos
    93: 
    94: 
    95: def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    96:     torch.nn.init.orthogonal_(layer.weight, std)
    97:     torch.nn.init.constant_(layer.bias, bias_const)
    98:     return layer
    99: 
   100: 
   101: def last_frame(obs: torch.Tensor) -> torch.Tensor:
   102:     return obs[:, 3:4, :, :].float()
   103: 
   104: 
   105: class RewardForwardFilter:
   106:     def __init__(self, gamma: float):
   107:         self.rewems = None
   108:         self.gamma = gamma
   109: 
   110:     def update(self, rews):
   111:         if self.rewems is None:
   112:             self.rewems = rews
   113:         else:
   114:             self.rewems = self.rewems * self.gamma + rews
   115:         return self.rewems
   116: 
   117: 
   118: class Agent(nn.Module):
   119:     def __init__(self, envs):
   120:         super().__init__()
   121:         self.network = nn.Sequential(
   122:             layer_init(nn.Conv2d(4, 32, 8, stride=4)),
   123:             nn.ReLU(),
   124:             layer_init(nn.Conv2d(32, 64, 4, stride=2)),
   125:             nn.ReLU(),
   126:             layer_init(nn.Conv2d(64, 64, 3, stride=1)),
   127:             nn.ReLU(),
   128:             nn.Flatten(),
   129:             layer_init(nn.Linear(64 * 7 * 7, 256)),
   130:             nn.ReLU(),
   131:             layer_init(nn.Linear(256, 448)),
   132:             nn.ReLU(),
   133:         )
   134:         self.extra_layer = nn.Sequential(
   135:             layer_init(nn.Linear(448, 448), std=0.1),
   136:             nn.ReLU(),
   137:         )
   138:         self.actor = nn.Sequential(
   139:             layer_init(nn.Linear(448, 448), std=0.01),
   140:             nn.ReLU(),
   141:             layer_init(nn.Linear(448, envs.single_action_space.n), std=0.01),
   142:         )
   143:         self.critic_ext = layer_init(nn.Linear(448, 1), std=0.01)
   144:         self.critic_int = layer_init(nn.Linear(448, 1), std=0.01)
   145: 
   146:     def _hidden(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
   147:         hidden = self.network(obs / 255.0)
   148:         features = self.extra_layer(hidden)
   149:         return hidden, features
   150: 
   151:     def get_logits(self, obs: torch.Tensor) -> torch.Tensor:
   152:         hidden, _ = self._hidden(obs)
   153:         return self.actor(hidden)
   154: 
   155:     def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
   156:         hidden, features = self._hidden(obs)
   157:         logits = self.actor(hidden)
   158:         probs = Categorical(logits=logits)
   159:         if action is None:
   160:             action = probs.sample()
   161:         return (
   162:             action,
   163:             probs.log_prob(action),
   164:             probs.entropy(),
   165:             self.critic_ext(features + hidden),
   166:             self.critic_int(features + hidden),
   167:         )
   168: 
   169:     def get_value(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
   170:         hidden, features = self._hidden(obs)
   171:         return self.critic_ext(features + hidden), self.critic_int(features + hidden)
   172: 
   173:     def get_deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
   174:         return torch.argmax(self.get_logits(obs), dim=1)
   175: 
   176: 
   177: # =====================================================================
   178: # EDITABLE: intrinsic reward design
   179: # =====================================================================
   180: class IntrinsicBonusModule(nn.Module):
   181:     """Default baseline: no intrinsic reward."""
   182: 
   183:     def __init__(self, action_dim: int, device: torch.device, args: Args):
   184:         super().__init__()
   185:         self.action_dim = action_dim
   186:         self.device = device
   187:         self.args = args
   188: 
   189:     def initialize(self, envs) -> None:
   190:         return None
   191: 
   192:     def trainable_parameters(self):
   193:         return []
   194: 
   195:     def update_batch_stats(self, batch_obs: torch.Tensor, batch_next_obs: torch.Tensor) -> None:
   196:         return None
   197: 
   198:     def compute_bonus(
   199:         self,
   200:         obs: torch.Tensor,
   201:         next_obs: torch.Tensor,
   202:         actions: torch.Tensor,
   203:     ) -> torch.Tensor:
   204:         return torch.zeros(obs.shape[0], device=self.device)
   205: 
   206:     def normalize_rollout_rewards(self, rollout_intrinsic: torch.Tensor) -> torch.Tensor:
   207:         return torch.zeros_like(rollout_intrinsic)
   208: 
   209:     def loss(
   210:         self,
   211:         batch_obs: torch.Tensor,
   212:         batch_next_obs: torch.Tensor,
   213:         batch_actions: torch.Tensor,
   214:     ) -> torch.Tensor:
   215:         return torch.zeros((), device=self.device)
   216: 
   217: 
   218: def mix_advantages(ext_advantages: torch.Tensor, int_advantages: torch.Tensor, args: Args) -> torch.Tensor:
   219:     return args.ext_coef * ext_advantages
   220: 
   221: 
   222: # =====================================================================
   223: # FIXED: evaluation and training loop
   224: # =====================================================================
   225: @torch.no_grad()
   226: def evaluate_policy(args: Args, agent: Agent, device: torch.device, seed: int) -> tuple[float, float]:
   227:     # Cap Atari evaluation episodes so deterministic no-op / survival loops cannot
   228:     # stall the whole benchmark at the first eval checkpoint.
   229:     envs = envpool.make(
   230:         args.env_id,
   231:         env_type="gym",
   232:         num_envs=1,
   233:         episodic_life=False,
   234:         reward_clip=True,
   235:         repeat_action_probability=0.25,
   236:         seed=seed,
   237:     )
   238:     envs.num_envs = 1
   239:     envs.single_action_space = envs.action_space
   240:     envs.single_observation_space = envs.observation_space
   241:     envs = RecordEpisodeStatistics(envs)
   242: 
   243:     returns = []
   244:     obs = torch.tensor(envs.reset(), device=device)
   245:     episode_steps = 0
   246:     while len(returns) < args.eval_episodes:
   247:         action = agent.get_deterministic_action(obs)
   248:         next_obs, _, done, info = envs.step(action.cpu().numpy())
   249:         obs = torch.tensor(next_obs, device=device)
   250:         episode_steps += 1
   251:         if done[0] or episode_steps >= args.eval_max_episode_steps:
   252:             returns.append(float(info["r"][0]))
   253:             episode_steps = 0
   254:             if len(returns) < args.eval_episodes:
   255:                 obs = torch.tensor(envs.reset(), device=device)
   256: 
   257:     envs.close()
   258:     returns_np = np.asarray(returns, dtype=np.float32)
   259:     return float(returns_np.mean()), float((returns_np != 0.0).mean())
   260: 
   261: 
   262: if __name__ == "__main__":
   263:     args = tyro.cli(Args)
   264:     args.batch_size = int(args.num_envs * args.num_steps)
   265:     args.minibatch_size = int(args.batch_size // args.num_minibatches)
   266:     args.num_iterations = args.total_timesteps // args.batch_size
   267: 
   268:     random.seed(args.seed)
   269:     np.random.seed(args.seed)
   270:     torch.manual_seed(args.seed)
   271:     torch.backends.cudnn.deterministic = args.torch_deterministic
   272: 
   273:     device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
   274: 
   275:     envs = envpool.make(
   276:         args.env_id,
   277:         env_type="gym",
   278:         num_envs=args.num_envs,
   279:         episodic_life=True,
   280:         reward_clip=True,
   281:         repeat_action_probability=0.25,
   282:         seed=args.seed,
   283:     )
   284:     envs.num_envs = args.num_envs
   285:     envs.single_action_space = envs.action_space
   286:     envs.single_observation_space = envs.observation_space
   287:     envs = RecordEpisodeStatistics(envs)
   288:     assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"
   289: 
   290:     agent = Agent(envs).to(device)
   291:     bonus_module = IntrinsicBonusModule(envs.single_action_space.n, device, args).to(device)
   292:     bonus_params = list(bonus_module.trainable_parameters())
   293:     optimizer = optim.Adam(list(agent.parameters()) + bonus_params, lr=args.learning_rate, eps=1e-5)
   294: 
   295:     obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
   296:     next_obs_buf = torch.zeros_like(obs)
   297:     actions = torch.zeros((args.num_steps, args.num_envs), device=device, dtype=torch.int64)
   298:     logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
   299:     rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
   300:     int_rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
   301:     dones = torch.zeros((args.num_steps, args.num_envs), device=device)
   302:     ext_values = torch.zeros((args.num_steps, args.num_envs), device=device)
   303:     int_values = torch.zeros((args.num_steps, args.num_envs), device=device)
   304: 
   305:     recent_returns: deque[float] = deque(maxlen=20)
   306:     eval_steps: list[int] = []
   307:     eval_returns: list[float] = []
   308:     eval_nonzero_rates: list[float] = []
   309:     next_eval_step = args.eval_interval
   310: 
   311:     global_step = 0
   312:     start_time = time.time()
   313:     # Reset once before any bootstrap rollout so the wrapper's episode buffers exist,
   314:     # then reset again to start actual training from a clean state.
   315:     envs.reset()
   316:     bonus_module.initialize(envs)
   317:     next_obs = torch.tensor(envs.reset(), device=device)
   318:     next_done = torch.zeros(args.num_envs, device=device)
   319:     eval_seed = args.seed + 10_000
   320: 
   321:     for iteration in range(1, args.num_iterations + 1):
   322:         if args.anneal_lr:
   323:             frac = 1.0 - (iteration - 1.0) / args.num_iterations
   324:             optimizer.param_groups[0]["lr"] = frac * args.learning_rate
   325: 
   326:         for step in range(args.num_steps):
   327:             global_step += args.num_envs
   328:             obs[step] = next_obs
   329:             dones[step] = next_done
   330: 
   331:             with torch.no_grad():
   332:                 value_ext, value_int = agent.get_value(obs[step])
   333:                 ext_values[step] = value_ext.flatten()
   334:                 int_values[step] = value_int.flatten()
   335:                 action, logprob, _, _, _ = agent.get_action_and_value(obs[step])
   336: 
   337:             actions[step] = action
   338:             logprobs[step] = logprob
   339: 
   340:             stepped_obs, reward, done, info = envs.step(action.cpu().numpy())
   341:             next_obs = torch.tensor(stepped_obs, device=device)
   342:             next_done = torch.tensor(done, device=device, dtype=torch.float32)
   343:             next_obs_buf[step] = next_obs
   344:             rewards[step] = torch.tensor(reward, device=device).view(-1)
   345:             with torch.no_grad():
   346:                 rollout_bonus = bonus_module.compute_bonus(obs[step], next_obs, action)
   347:             int_rewards[step] = rollout_bonus * (1.0 - next_done)
   348: 
   349:             for idx, terminated in enumerate(done):
   350:                 if terminated and info["lives"][idx] == 0:
   351:                     recent_returns.append(float(info["r"][idx]))
   352: 
   353:         int_rewards = bonus_module.normalize_rollout_rewards(int_rewards)
   354: 
   355:         with torch.no_grad():
   356:             next_value_ext, next_value_int = agent.get_value(next_obs)
   357:             next_value_ext = next_value_ext.reshape(1, -1)
   358:             next_value_int = next_value_int.reshape(1, -1)
   359:             ext_advantages = torch.zeros_like(rewards, device=device)
   360:             int_advantages = torch.zeros_like(int_rewards, device=device)
   361:             ext_lastgaelam = 0
   362:             int_lastgaelam = 0
   363:             for t in reversed(range(args.num_steps)):
   364:                 if t == args.num_steps - 1:
   365:                     ext_nextnonterminal = 1.0 - next_done
   366:                     int_nextnonterminal = ext_nextnonterminal
   367:                     ext_nextvalues = next_value_ext
   368:                     int_nextvalues = next_value_int
   369:                 else:
   370:                     ext_nextnonterminal = 1.0 - dones[t + 1]
   371:                     int_nextnonterminal = ext_nextnonterminal
   372:                     ext_nextvalues = ext_values[t + 1]
   373:                     int_nextvalues = int_values[t + 1]
   374:                 ext_delta = rewards[t] + args.gamma * ext_nextvalues * ext_nextnonterminal - ext_values[t]
   375:                 int_delta = int_rewards[t] + args.int_gamma * int_nextvalues * int_nextnonterminal - int_values[t]
   376:                 ext_advantages[t] = ext_lastgaelam = (
   377:                     ext_delta + args.gamma * args.gae_lambda * ext_nextnonterminal * ext_lastgaelam
   378:                 )
   379:                 int_advantages[t] = int_lastgaelam = (
   380:                     int_delta + args.int_gamma * args.gae_lambda * int_nextnonterminal * int_lastgaelam
   381:                 )
   382:             ext_returns = ext_advantages + ext_values
   383:             int_returns = int_advantages + int_values
   384: 
   385:         b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
   386:         b_next_obs = next_obs_buf.reshape((-1,) + envs.single_observation_space.shape)
   387:         b_actions = actions.reshape(-1)
   388:         b_logprobs = logprobs.reshape(-1)
   389:         b_ext_advantages = ext_advantages.reshape(-1)
   390:         b_int_advantages = int_advantages.reshape(-1)
   391:         b_ext_returns = ext_returns.reshape(-1)
   392:         b_int_returns = int_returns.reshape(-1)
   393:         b_ext_values = ext_values.reshape(-1)
   394: 
   395:         b_advantages = mix_advantages(b_ext_advantages, b_int_advantages, args)
   396:         bonus_module.update_batch_stats(b_obs, b_next_obs)
   397: 
   398:         b_inds = np.arange(args.batch_size)
   399:         clipfracs = []
   400:         for epoch in range(args.update_epochs):
   401:             np.random.shuffle(b_inds)
   402:             for start in range(0, args.batch_size, args.minibatch_size):
   403:                 end = start + args.minibatch_size
   404:                 mb_inds = b_inds[start:end]
   405: 
   406:                 _, newlogprob, entropy, new_ext_values, new_int_values = agent.get_action_and_value(
   407:                     b_obs[mb_inds],
   408:                     b_actions[mb_inds],
   409:                 )
   410:                 logratio = newlogprob - b_logprobs[mb_inds]
   411:                 ratio = logratio.exp()
   412: 
   413:                 with torch.no_grad():
   414:                     old_approx_kl = (-logratio).mean()
   415:                     approx_kl = ((ratio - 1) - logratio).mean()
   416:                     clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())
   417: 
   418:                 mb_advantages = b_advantages[mb_inds]
   419:                 if args.norm_adv:
   420:                     mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
   421: 
   422:                 pg_loss1 = -mb_advantages * ratio
   423:                 pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
   424:                 pg_loss = torch.max(pg_loss1, pg_loss2).mean()
   425: 
   426:                 new_ext_values = new_ext_values.view(-1)
   427:                 new_int_values = new_int_values.view(-1)
   428:                 if args.clip_vloss:
   429:                     ext_v_loss_unclipped = (new_ext_values - b_ext_returns[mb_inds]) ** 2
   430:                     ext_v_clipped = b_ext_values[mb_inds] + torch.clamp(
   431:                         new_ext_values - b_ext_values[mb_inds],
   432:                         -args.clip_coef,
   433:                         args.clip_coef,
   434:                     )
   435:                     ext_v_loss_clipped = (ext_v_clipped - b_ext_returns[mb_inds]) ** 2
   436:                     ext_v_loss = 0.5 * torch.max(ext_v_loss_unclipped, ext_v_loss_clipped).mean()
   437:                 else:
   438:                     ext_v_loss = 0.5 * ((new_ext_values - b_ext_returns[mb_inds]) ** 2).mean()
   439:                 int_v_loss = 0.5 * ((new_int_values - b_int_returns[mb_inds]) ** 2).mean()
   440:                 v_loss = ext_v_loss + int_v_loss
   441: 
   442:                 entropy_loss = entropy.mean()
   443:                 bonus_loss = bonus_module.loss(
   444:                     b_obs[mb_inds],
   445:                     b_next_obs[mb_inds],
   446:                     b_actions[mb_inds],
   447:                 )
   448:                 loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss + bonus_loss
   449: 
   450:                 optimizer.zero_grad()
   451:                 loss.backward()
   452:                 nn.utils.clip_grad_norm_(list(agent.parameters()) + bonus_params, args.max_grad_norm)
   453:                 optimizer.step()
   454: 
   455:             if args.target_kl is not None and approx_kl > args.target_kl:
   456:                 break
   457: 
   458:         latest_eval_return = float("nan")
   459:         latest_nonzero = float("nan")
   460:         if global_step >= next_eval_step or iteration == args.num_iterations:
   461:             latest_eval_return, latest_nonzero = evaluate_policy(args, agent, device, eval_seed)
   462:             eval_steps.append(global_step)
   463:             eval_returns.append(latest_eval_return)
   464:             eval_nonzero_rates.append(latest_nonzero)
   465:             next_eval_step += args.eval_interval
   466: 
   467:         avg_return = float(np.mean(recent_returns)) if recent_returns else 0.0
   468:         avg_intrinsic = float(int_rewards.mean().item())
   469:         sps = int(global_step / max(time.time() - start_time, 1e-6))
   470:         print(
   471:             f"TRAIN_METRICS step={global_step} avg_return={avg_return:.4f} "
   472:             f"avg_intrinsic={avg_intrinsic:.6f} eval_return={latest_eval_return:.4f} "
   473:             f"nonzero_rate={latest_nonzero:.4f} sps={sps}",
   474:             flush=True,
   475:         )
   476: 
   477:     if not eval_returns:
   478:         final_eval_return, final_nonzero = evaluate_policy(args, agent, device, eval_seed)
   479:         eval_steps.append(global_step)
   480:         eval_returns.append(final_eval_return)
   481:         eval_nonzero_rates.append(final_nonzero)
   482: 
   483:     auc = float(np.trapz(np.asarray(eval_returns, dtype=np.float32), np.asarray(eval_steps, dtype=np.float32)))
   484:     auc /= max(float(eval_steps[-1]), 1.0)
   485:     final_eval_return = float(eval_returns[-1])
   486:     final_nonzero = float(eval_nonzero_rates[-1])
   487:     best_eval_return = float(np.max(eval_returns))
   488:     print(
   489:         f"TEST_METRICS eval_return={final_eval_return:.4f} auc={auc:.6f} "
   490:         f"nonzero_rate={final_nonzero:.4f} best_eval_return={best_eval_return:.4f}",
   491:         flush=True,
   492:     )
```




## How You Will Be Evaluated

After you finish, evaluation runs a fixed set of scripts and aggregates the
metrics they emit. These scripts are **not** in your workspace — you cannot
read or modify them. The labels below indicate what each evaluation tests:

- **tutankham-v5** — wall-clock budget `08:00:00`, compute share `1.0`
- **frostbite-v5** — wall-clock budget `08:00:00`, compute share `1.0`
- **private-eye-v5** — wall-clock budget `08:00:00`, compute share `1.0`


Scoring uses the same `combined_score` aggregation as the MLS-Bench
leaderboard. Multiple seeds are averaged.



## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `ppo` baseline — editable region  [READ-ONLY — reference implementation]

In `cleanrl/cleanrl/custom_intrinsic_exploration.py`:

```python
Lines 179–218:
   176: 
   177: # =====================================================================
   178: # EDITABLE: intrinsic reward design
   179: class IntrinsicBonusModule(nn.Module):
   180:     """Baseline: no intrinsic reward."""
   181: 
   182:     def __init__(self, action_dim: int, device: torch.device, args: Args):
   183:         super().__init__()
   184:         self.action_dim = action_dim
   185:         self.device = device
   186:         self.args = args
   187: 
   188:     def initialize(self, envs) -> None:
   189:         return None
   190: 
   191:     def trainable_parameters(self):
   192:         return []
   193: 
   194:     def update_batch_stats(self, batch_obs: torch.Tensor, batch_next_obs: torch.Tensor) -> None:
   195:         return None
   196: 
   197:     def compute_bonus(
   198:         self,
   199:         obs: torch.Tensor,
   200:         next_obs: torch.Tensor,
   201:         actions: torch.Tensor,
   202:     ) -> torch.Tensor:
   203:         return torch.zeros(obs.shape[0], device=self.device)
   204: 
   205:     def normalize_rollout_rewards(self, rollout_intrinsic: torch.Tensor) -> torch.Tensor:
   206:         return torch.zeros_like(rollout_intrinsic)
   207: 
   208:     def loss(
   209:         self,
   210:         batch_obs: torch.Tensor,
   211:         batch_next_obs: torch.Tensor,
   212:         batch_actions: torch.Tensor,
   213:     ) -> torch.Tensor:
   214:         return torch.zeros((), device=self.device)
   215: 
   216: 
   217: def mix_advantages(ext_advantages: torch.Tensor, int_advantages: torch.Tensor, args: Args) -> torch.Tensor:
   218:     return args.ext_coef * ext_advantages
   219: 
   220: 
   221: # =====================================================================
```

### `rnd` baseline — editable region  [READ-ONLY — reference implementation]

In `cleanrl/cleanrl/custom_intrinsic_exploration.py`:

```python
Lines 179–281:
   176: 
   177: # =====================================================================
   178: # EDITABLE: intrinsic reward design
   179: class IntrinsicBonusModule(nn.Module):
   180:     """Random Network Distillation intrinsic bonus."""
   181: 
   182:     def __init__(self, action_dim: int, device: torch.device, args: Args):
   183:         super().__init__()
   184:         self.action_dim = action_dim
   185:         self.device = device
   186:         self.args = args
   187:         self.obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))
   188:         self.reward_rms = RunningMeanStd()
   189:         self.discounted_reward = RewardForwardFilter(args.int_gamma)
   190: 
   191:         feature_output = 7 * 7 * 64
   192:         self.predictor = nn.Sequential(
   193:             layer_init(nn.Conv2d(1, 32, 8, stride=4)),
   194:             nn.LeakyReLU(),
   195:             layer_init(nn.Conv2d(32, 64, 4, stride=2)),
   196:             nn.LeakyReLU(),
   197:             layer_init(nn.Conv2d(64, 64, 3, stride=1)),
   198:             nn.LeakyReLU(),
   199:             nn.Flatten(),
   200:             layer_init(nn.Linear(feature_output, 512)),
   201:             nn.ReLU(),
   202:             layer_init(nn.Linear(512, 512)),
   203:             nn.ReLU(),
   204:             layer_init(nn.Linear(512, 512)),
   205:         )
   206:         self.target = nn.Sequential(
   207:             layer_init(nn.Conv2d(1, 32, 8, stride=4)),
   208:             nn.LeakyReLU(),
   209:             layer_init(nn.Conv2d(32, 64, 4, stride=2)),
   210:             nn.LeakyReLU(),
   211:             layer_init(nn.Conv2d(64, 64, 3, stride=1)),
   212:             nn.LeakyReLU(),
   213:             nn.Flatten(),
   214:             layer_init(nn.Linear(feature_output, 512)),
   215:         )
   216:         for param in self.target.parameters():
   217:             param.requires_grad = False
   218: 
   219:     def initialize(self, envs) -> None:
   220:         bootstrap = []
   221:         total_steps = self.args.num_steps * self.args.num_iterations_obs_norm_init
   222:         for _ in range(total_steps):
   223:             random_actions = np.random.randint(0, envs.single_action_space.n, size=(self.args.num_envs,))
   224:             sampled_obs, _, _, _ = envs.step(random_actions)
   225:             bootstrap.append(sampled_obs[:, 3:4, :, :])
   226:             if len(bootstrap) >= self.args.num_steps:
   227:                 stacked = np.concatenate(bootstrap, axis=0)
   228:                 self.obs_rms.update(stacked)
   229:                 bootstrap.clear()
   230: 
   231:     def trainable_parameters(self):
   232:         return list(self.predictor.parameters())
   233: 
   234:     def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
   235:         mean = torch.from_numpy(self.obs_rms.mean).to(self.device)
   236:         var = torch.from_numpy(self.obs_rms.var).to(self.device)
   237:         return ((last_frame(obs) - mean) / torch.sqrt(var)).clip(-5, 5).float()
   238: 
   239:     def update_batch_stats(self, batch_obs: torch.Tensor, batch_next_obs: torch.Tensor) -> None:
   240:         self.obs_rms.update(last_frame(batch_next_obs).cpu().numpy())
   241: 
   242:     def compute_bonus(
   243:         self,
   244:         obs: torch.Tensor,
   245:         next_obs: torch.Tensor,
   246:         actions: torch.Tensor,
   247:     ) -> torch.Tensor:
   248:         norm_next = self._normalize_obs(next_obs)
   249:         target_feature = self.target(norm_next)
   250:         predict_feature = self.predictor(norm_next)
   251:         return ((target_feature - predict_feature).pow(2).sum(1) / 2).detach()
   252: 
   253:     def normalize_rollout_rewards(self, rollout_intrinsic: torch.Tensor) -> torch.Tensor:
   254:         discounted = np.stack(
   255:             [self.discounted_reward.update(reward_per_step) for reward_per_step in rollout_intrinsic.cpu().numpy()],
   256:             axis=0,
   257:         )
   258:         flat_discounted = discounted.reshape(-1)
   259:         self.reward_rms.update_from_moments(
   260:             float(flat_discounted.mean()),
   261:             float(flat_discounted.var()),
   262:             int(flat_discounted.size),
   263:         )
   264:         return rollout_intrinsic / float(np.sqrt(self.reward_rms.var + 1e-8))
   265: 
   266:     def loss(
   267:         self,
   268:         batch_obs: torch.Tensor,
   269:         batch_next_obs: torch.Tensor,
   270:         batch_actions: torch.Tensor,
   271:     ) -> torch.Tensor:
   272:         norm_next = self._normalize_obs(batch_next_obs)
   273:         predict_feature = self.predictor(norm_next)
   274:         target_feature = self.target(norm_next).detach()
   275:         forward_loss = F.mse_loss(predict_feature, target_feature, reduction="none").mean(-1)
   276:         mask = (torch.rand(len(forward_loss), device=self.device) < self.args.update_proportion).float()
   277:         return (forward_loss * mask).sum() / torch.clamp(mask.sum(), min=1.0)
   278: 
   279: 
   280: def mix_advantages(ext_advantages: torch.Tensor, int_advantages: torch.Tensor, args: Args) -> torch.Tensor:
   281:     return args.ext_coef * ext_advantages + args.int_coef * int_advantages
   282: 
   283: 
   284: # =====================================================================
```

### `icm` baseline — editable region  [READ-ONLY — reference implementation]

In `cleanrl/cleanrl/custom_intrinsic_exploration.py`:

```python
Lines 179–280:
   176: 
   177: # =====================================================================
   178: # EDITABLE: intrinsic reward design
   179: class IntrinsicBonusModule(nn.Module):
   180:     """Intrinsic Curiosity Module baseline."""
   181: 
   182:     def __init__(self, action_dim: int, device: torch.device, args: Args):
   183:         super().__init__()
   184:         self.action_dim = action_dim
   185:         self.device = device
   186:         self.args = args
   187:         self.obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))
   188:         self.reward_rms = RunningMeanStd()
   189:         self.discounted_reward = RewardForwardFilter(args.int_gamma)
   190: 
   191:         feature_output = 7 * 7 * 64
   192:         self.encoder = nn.Sequential(
   193:             layer_init(nn.Conv2d(1, 32, 8, stride=4)),
   194:             nn.ReLU(),
   195:             layer_init(nn.Conv2d(32, 64, 4, stride=2)),
   196:             nn.ReLU(),
   197:             layer_init(nn.Conv2d(64, 64, 3, stride=1)),
   198:             nn.ReLU(),
   199:             nn.Flatten(),
   200:             layer_init(nn.Linear(feature_output, 256)),
   201:             nn.ReLU(),
   202:         )
   203:         self.inverse_model = nn.Sequential(
   204:             layer_init(nn.Linear(512, 256)),
   205:             nn.ReLU(),
   206:             layer_init(nn.Linear(256, action_dim), std=0.01),
   207:         )
   208:         self.forward_model = nn.Sequential(
   209:             layer_init(nn.Linear(256 + action_dim, 256)),
   210:             nn.ReLU(),
   211:             layer_init(nn.Linear(256, 256)),
   212:         )
   213: 
   214:     def initialize(self, envs) -> None:
   215:         bootstrap = []
   216:         total_steps = self.args.num_steps * self.args.num_iterations_obs_norm_init
   217:         for _ in range(total_steps):
   218:             random_actions = np.random.randint(0, envs.single_action_space.n, size=(self.args.num_envs,))
   219:             sampled_obs, _, _, _ = envs.step(random_actions)
   220:             bootstrap.append(sampled_obs[:, 3:4, :, :])
   221:             if len(bootstrap) >= self.args.num_steps:
   222:                 stacked = np.concatenate(bootstrap, axis=0)
   223:                 self.obs_rms.update(stacked)
   224:                 bootstrap.clear()
   225: 
   226:     def trainable_parameters(self):
   227:         return list(self.parameters())
   228: 
   229:     def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
   230:         mean = torch.from_numpy(self.obs_rms.mean).to(self.device)
   231:         var = torch.from_numpy(self.obs_rms.var).to(self.device)
   232:         return ((last_frame(obs) - mean) / torch.sqrt(var)).clip(-5, 5).float()
   233: 
   234:     def _one_hot(self, actions: torch.Tensor) -> torch.Tensor:
   235:         return F.one_hot(actions.long(), num_classes=self.action_dim).float()
   236: 
   237:     def update_batch_stats(self, batch_obs: torch.Tensor, batch_next_obs: torch.Tensor) -> None:
   238:         self.obs_rms.update(last_frame(batch_next_obs).cpu().numpy())
   239: 
   240:     def compute_bonus(
   241:         self,
   242:         obs: torch.Tensor,
   243:         next_obs: torch.Tensor,
   244:         actions: torch.Tensor,
   245:     ) -> torch.Tensor:
   246:         obs_feat = self.encoder(self._normalize_obs(obs))
   247:         next_feat = self.encoder(self._normalize_obs(next_obs))
   248:         pred_next_feat = self.forward_model(torch.cat([obs_feat, self._one_hot(actions)], dim=1))
   249:         return 0.5 * (pred_next_feat - next_feat).pow(2).mean(dim=1).detach()
   250: 
   251:     def normalize_rollout_rewards(self, rollout_intrinsic: torch.Tensor) -> torch.Tensor:
   252:         discounted = np.stack(
   253:             [self.discounted_reward.update(reward_per_step) for reward_per_step in rollout_intrinsic.cpu().numpy()],
   254:             axis=0,
   255:         )
   256:         flat_discounted = discounted.reshape(-1)
   257:         self.reward_rms.update_from_moments(
   258:             float(flat_discounted.mean()),
   259:             float(flat_discounted.var()),
   260:             int(flat_discounted.size),
   261:         )
   262:         return rollout_intrinsic / float(np.sqrt(self.reward_rms.var + 1e-8))
   263: 
   264:     def loss(
   265:         self,
   266:         batch_obs: torch.Tensor,
   267:         batch_next_obs: torch.Tensor,
   268:         batch_actions: torch.Tensor,
   269:     ) -> torch.Tensor:
   270:         obs_feat = self.encoder(self._normalize_obs(batch_obs))
   271:         next_feat = self.encoder(self._normalize_obs(batch_next_obs))
   272:         pred_next_feat = self.forward_model(torch.cat([obs_feat, self._one_hot(batch_actions)], dim=1))
   273:         pred_action = self.inverse_model(torch.cat([obs_feat, next_feat], dim=1))
   274:         inverse_loss = F.cross_entropy(pred_action, batch_actions.long())
   275:         forward_loss = 0.5 * (pred_next_feat - next_feat.detach()).pow(2).mean()
   276:         return inverse_loss + 0.2 * forward_loss
   277: 
   278: 
   279: def mix_advantages(ext_advantages: torch.Tensor, int_advantages: torch.Tensor, args: Args) -> torch.Tensor:
   280:     return args.ext_coef * ext_advantages + args.int_coef * int_advantages
   281: 
   282: 
   283: # =====================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
