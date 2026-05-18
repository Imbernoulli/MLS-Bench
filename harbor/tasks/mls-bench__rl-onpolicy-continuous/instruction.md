# MLS-Bench: rl-onpolicy-continuous

# Online RL: On-Policy Actor-Critic for Continuous Control

## Research Question
Design and implement an on-policy actor-critic RL algorithm for
continuous control. Your code goes in `custom_onpolicy_continuous.py`.
Several reference implementations are provided as read-only `*.edit.py`
baselines.

## Background
On-policy methods collect trajectories with the current policy, compute
advantages with Generalized Advantage Estimation (GAE), and update the
policy via mini-batch optimization on freshly collected data. Compared
to off-policy methods they avoid replay-distribution mismatch but are
less sample-efficient and more sensitive to update stability. Different
design points address these tensions through clipped surrogate
objectives, adaptive penalties, stochasticity injection,
advantage-weighted regression, or other policy-update rules.

Reference baselines spanning the design space:
- **PPO (clip)** — Schulman et al., "Proximal Policy Optimization
  Algorithms" (arXiv:1707.06347). Clipped surrogate with default clip
  range `epsilon = 0.2` and GAE `lambda = 0.95`.
- **PPO-Penalty** — KL-penalty variant from the same paper.
- **AWR** — Peng et al., "Advantage-Weighted Regression: Simple and
  Scalable Off-Policy Reinforcement Learning" (arXiv:1910.00177).
  Advantage-weighted supervised policy update with default temperature
  `beta = 1.0`.

## Constraints
- Network architecture dimensions are FIXED and cannot be modified.
- Total parameter count is enforced at runtime; the contribution must
  be algorithmic (action distribution, surrogate loss, penalty,
  exploration injection, value loss) rather than capacity.
- Do **not** simply copy a reference implementation with minor changes.

## Evaluation
Trained and evaluated on Gymnasium MuJoCo continuous-control
environments including HalfCheetah-v4, Hopper-v4 and Walker2d-v4 within
a fixed interaction budget. Metric: mean episodic return over
evaluation episodes (higher is better). Strong methods should remain
reliable across environments with different dynamics rather than tuning
to one.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/cleanrl/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to score zero.

- `cleanrl/cleanrl/custom_onpolicy_continuous.py`
- editable lines **175–221**




## Readable Context


### `cleanrl/cleanrl/custom_onpolicy_continuous.py`  [EDITABLE — lines 175–221 only]

```python
     1: # Custom on-policy continuous RL algorithm for MLS-Bench
     2: #
     3: # FIXED sections: config, env, utilities, network architecture, training loop.
     4: # EDITABLE section: get_action_and_value method and compute_losses function.
     5: import copy
     6: import os
     7: import random
     8: import time
     9: from dataclasses import dataclass
    10: 
    11: import gymnasium as gym
    12: import numpy as np
    13: import torch
    14: import torch.nn as nn
    15: import torch.nn.functional as F
    16: import torch.optim as optim
    17: import tyro
    18: from torch.distributions.normal import Normal
    19: 
    20: 
    21: # =====================================================================
    22: # FIXED: Configuration
    23: # =====================================================================
    24: @dataclass
    25: class Args:
    26:     exp_name: str = os.path.basename(__file__)[: -len(".py")]
    27:     """the name of this experiment"""
    28:     seed: int = 1
    29:     """seed of the experiment"""
    30:     torch_deterministic: bool = True
    31:     """if toggled, `torch.backends.cudnn.deterministic=False`"""
    32:     cuda: bool = True
    33:     """if toggled, cuda will be enabled by default"""
    34: 
    35:     # Algorithm specific arguments
    36:     env_id: str = "HalfCheetah-v4"
    37:     """the id of the environment"""
    38:     total_timesteps: int = 1000000
    39:     """total timesteps of the experiments"""
    40:     learning_rate: float = 3e-4
    41:     """the learning rate of the optimizer"""
    42:     num_envs: int = 1
    43:     """the number of parallel game environments"""
    44:     num_steps: int = 2048
    45:     """the number of steps to run in each environment per policy rollout"""
    46:     anneal_lr: bool = True
    47:     """Toggle learning rate annealing for policy and value networks"""
    48:     gamma: float = 0.99
    49:     """the discount factor gamma"""
    50:     gae_lambda: float = 0.95
    51:     """the lambda for the general advantage estimation"""
    52:     num_minibatches: int = 32
    53:     """the number of mini-batches"""
    54:     update_epochs: int = 10
    55:     """the K epochs to update the policy"""
    56:     norm_adv: bool = True
    57:     """Toggles advantages normalization"""
    58:     clip_coef: float = 0.2
    59:     """the surrogate clipping coefficient"""
    60:     clip_vloss: bool = True
    61:     """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    62:     ent_coef: float = 0.0
    63:     """coefficient of the entropy"""
    64:     vf_coef: float = 0.5
    65:     """coefficient of the value function"""
    66:     max_grad_norm: float = 0.5
    67:     """the maximum norm for the gradient clipping"""
    68:     target_kl: float = None
    69:     """the target KL divergence threshold"""
    70:     eval_freq: int = 50000
    71:     """evaluation frequency (timesteps)"""
    72:     eval_episodes: int = 10
    73:     """number of evaluation episodes"""
    74: 
    75:     # to be filled in runtime
    76:     batch_size: int = 0
    77:     """the batch size (computed in runtime)"""
    78:     minibatch_size: int = 0
    79:     """the mini-batch size (computed in runtime)"""
    80:     num_iterations: int = 0
    81:     """the number of iterations (computed in runtime)"""
    82: 
    83: 
    84: # =====================================================================
    85: # FIXED: Environment setup
    86: # =====================================================================
    87: def make_env(env_id, idx, gamma):
    88:     def thunk():
    89:         env = gym.make(env_id)
    90:         env = gym.wrappers.FlattenObservation(env)
    91:         env = gym.wrappers.RecordEpisodeStatistics(env)
    92:         env = gym.wrappers.ClipAction(env)
    93:         env = gym.wrappers.NormalizeObservation(env)
    94:         env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    95:         env = gym.wrappers.NormalizeReward(env, gamma=gamma)
    96:         env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    97:         return env
    98:     return thunk
    99: 
   100: 
   101: # =====================================================================
   102: # FIXED: Utilities
   103: # =====================================================================
   104: def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
   105:     torch.nn.init.orthogonal_(layer.weight, std)
   106:     torch.nn.init.constant_(layer.bias, bias_const)
   107:     return layer
   108: 
   109: 
   110: @torch.no_grad()
   111: def eval_agent(env_id, agent, device, n_episodes, seed, gamma=0.99, obs_rms=None):
   112:     """Evaluate agent for n_episodes using same env wrappers as training.
   113:     If obs_rms provided, copies training normalization stats to eval env.
   114:     Returns array of raw (un-normalized) episode returns."""
   115:     eval_envs = gym.vector.SyncVectorEnv(
   116:         [make_env(env_id, 0, gamma)]
   117:     )
   118:     if obs_rms is not None:
   119:         _eval_env = eval_envs.envs[0]
   120:         while hasattr(_eval_env, 'env'):
   121:             if isinstance(_eval_env, gym.wrappers.NormalizeObservation):
   122:                 _eval_env.obs_rms = copy.deepcopy(obs_rms)
   123:                 break
   124:             _eval_env = _eval_env.env
   125:     episode_rewards = []
   126:     obs, _ = eval_envs.reset(seed=seed)
   127:     while len(episode_rewards) < n_episodes:
   128:         obs_t = torch.Tensor(obs).to(device)
   129:         action, _, _, _ = agent.get_action_and_value(obs_t)
   130:         obs, _, _, _, infos = eval_envs.step(action.cpu().numpy())
   131:         if "final_info" in infos:
   132:             for info in infos["final_info"]:
   133:                 if info and "episode" in info:
   134:                     episode_rewards.append(float(info["episode"]["r"]))
   135:     eval_envs.close()
   136:     return np.asarray(episode_rewards)
   137: 
   138: 
   139: # =====================================================================
   140: # FIXED: Agent architecture (network capacity is fixed)
   141: # =====================================================================
   142: class Agent(nn.Module):
   143:     """On-policy actor-critic agent.
   144: 
   145:     Architecture is FIXED (2x64 MLP for both actor and critic).
   146:     Only get_action_and_value is editable — this is where algorithmic
   147:     innovation happens (distribution type, squashing, etc.).
   148:     """
   149: 
   150:     def __init__(self, obs_dim, action_dim):
   151:         super().__init__()
   152:         h = 64
   153:         self.critic = nn.Sequential(
   154:             nn.Linear(obs_dim, h),
   155:             nn.Tanh(),
   156:             nn.Linear(h, h),
   157:             nn.Tanh(),
   158:             nn.Linear(h, 1),
   159:         )
   160:         self.actor_mean = nn.Sequential(
   161:             nn.Linear(obs_dim, h),
   162:             nn.Tanh(),
   163:             nn.Linear(h, h),
   164:             nn.Tanh(),
   165:             nn.Linear(h, action_dim),
   166:         )
   167:         self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
   168: 
   169:     def get_value(self, obs):
   170:         return self.critic(obs)
   171: 
   172:     # =================================================================
   173:     # EDITABLE: get_action_and_value and compute_losses
   174:     # =================================================================
   175:     def get_action_and_value(self, obs, action=None):
   176:         action_mean = self.actor_mean(obs)
   177:         action_logstd = self.actor_logstd.expand_as(action_mean)
   178:         action_std = torch.exp(action_logstd)
   179:         probs = Normal(action_mean, action_std)
   180:         if action is None:
   181:             action = probs.sample()
   182:         return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)
   183: 
   184: 
   185: def compute_losses(agent, mb_obs, mb_actions, mb_logprobs, mb_advantages, mb_returns, mb_values, args):
   186:     """Compute policy and value losses for a minibatch.
   187: 
   188:     Args:
   189:         agent: the Agent instance
   190:         mb_obs: minibatch observations
   191:         mb_actions: minibatch actions
   192:         mb_logprobs: minibatch old log probabilities
   193:         mb_advantages: minibatch advantages
   194:         mb_returns: minibatch returns
   195:         mb_values: minibatch old values
   196:         args: hyperparameters
   197: 
   198:     Returns:
   199:         (loss, pg_loss, v_loss, entropy_loss, approx_kl, clipfrac)
   200:     """
   201:     _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions)
   202:     logratio = newlogprob - mb_logprobs
   203:     ratio = logratio.exp()
   204: 
   205:     with torch.no_grad():
   206:         approx_kl = ((ratio - 1) - logratio).mean()
   207:         clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
   208: 
   209:     # Policy loss -- placeholder
   210:     pg_loss = (-mb_advantages * ratio).mean()
   211: 
   212:     # Value loss -- placeholder
   213:     newvalue = newvalue.view(-1)
   214:     v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
   215: 
   216:     entropy_loss = entropy.mean()
   217:     loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
   218: 
   219:     return loss, pg_loss, v_loss, entropy_loss, approx_kl, clipfrac
   220: 
   221: 
   222: # =====================================================================
   223: # FIXED: Training loop
   224: # =====================================================================
   225: if __name__ == "__main__":
   226:     args = tyro.cli(Args)
   227:     args.batch_size = int(args.num_envs * args.num_steps)
   228:     args.minibatch_size = int(args.batch_size // args.num_minibatches)
   229:     args.num_iterations = args.total_timesteps // args.batch_size
   230:     run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
   231: 
   232:     # Seeding
   233:     random.seed(args.seed)
   234:     np.random.seed(args.seed)
   235:     torch.manual_seed(args.seed)
   236:     torch.backends.cudnn.deterministic = args.torch_deterministic
   237: 
   238:     device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
   239: 
   240:     # Environment setup
   241:     envs = gym.vector.SyncVectorEnv(
   242:         [make_env(args.env_id, i, args.gamma) for i in range(args.num_envs)]
   243:     )
   244:     assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
   245: 
   246:     obs_dim = np.array(envs.single_observation_space.shape).prod()
   247:     action_dim = np.prod(envs.single_action_space.shape)
   248: 
   249:     agent = Agent(obs_dim, action_dim).to(device)
   250: 
   251:     # Parameter count guard: prevent network capacity hacking
   252:     _param_count = sum(p.numel() for p in agent.parameters())
   253:     _expected_params = (obs_dim * 64 + 64) + (64 * 64 + 64) + (64 * 1 + 1) \
   254:                      + (obs_dim * 64 + 64) + (64 * 64 + 64) + (64 * action_dim + action_dim) \
   255:                      + action_dim  # actor_logstd
   256:     assert _param_count == _expected_params, (
   257:         f"Parameter count mismatch: got {_param_count}, expected {_expected_params}. "
   258:         f"Do not modify the network architecture — only get_action_and_value and compute_losses are editable."
   259:     )
   260: 
   261:     optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
   262: 
   263:     # Storage setup
   264:     obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
   265:     actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
   266:     logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
   267:     rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
   268:     dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
   269:     values = torch.zeros((args.num_steps, args.num_envs)).to(device)
   270: 
   271:     # Start the game
   272:     global_step = 0
   273:     start_time = time.time()
   274:     next_obs, _ = envs.reset(seed=args.seed)
   275:     next_obs = torch.Tensor(next_obs).to(device)
   276:     next_done = torch.zeros(args.num_envs).to(device)
   277: 
   278:     for iteration in range(1, args.num_iterations + 1):
   279:         # Annealing the rate if instructed to do so
   280:         if args.anneal_lr:
   281:             frac = 1.0 - (iteration - 1.0) / args.num_iterations
   282:             lrnow = frac * args.learning_rate
   283:             optimizer.param_groups[0]["lr"] = lrnow
   284: 
   285:         for step in range(0, args.num_steps):
   286:             global_step += args.num_envs
   287:             obs[step] = next_obs
   288:             dones[step] = next_done
   289: 
   290:             # Action logic
   291:             with torch.no_grad():
   292:                 action, logprob, _, value = agent.get_action_and_value(next_obs)
   293:                 values[step] = value.flatten()
   294:             actions[step] = action
   295:             logprobs[step] = logprob
   296: 
   297:             # Execute the game and log data
   298:             next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
   299:             next_done = np.logical_or(terminations, truncations)
   300:             rewards[step] = torch.tensor(reward).to(device).view(-1)
   301:             next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
   302: 
   303:             if "final_info" in infos:
   304:                 for info in infos["final_info"]:
   305:                     if info and "episode" in info:
   306:                         print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
   307: 
   308:         # Bootstrap value if not done
   309:         with torch.no_grad():
   310:             next_value = agent.get_value(next_obs).reshape(1, -1)
   311:             advantages = torch.zeros_like(rewards).to(device)
   312:             lastgaelam = 0
   313:             for t in reversed(range(args.num_steps)):
   314:                 if t == args.num_steps - 1:
   315:                     nextnonterminal = 1.0 - next_done
   316:                     nextvalues = next_value
   317:                 else:
   318:                     nextnonterminal = 1.0 - dones[t + 1]
   319:                     nextvalues = values[t + 1]
   320:                 delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
   321:                 advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
   322:             returns = advantages + values
   323: 
   324:         # Flatten the batch
   325:         b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
   326:         b_logprobs = logprobs.reshape(-1)
   327:         b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
   328:         b_advantages = advantages.reshape(-1)
   329:         b_returns = returns.reshape(-1)
   330:         b_values = values.reshape(-1)
   331: 
   332:         # Optimizing the policy and value network
   333:         b_inds = np.arange(args.batch_size)
   334:         clipfracs = []
   335:         for epoch in range(args.update_epochs):
   336:             np.random.shuffle(b_inds)
   337:             for start in range(0, args.batch_size, args.minibatch_size):
   338:                 end = start + args.minibatch_size
   339:                 mb_inds = b_inds[start:end]
   340: 
   341:                 if args.norm_adv:
   342:                     mb_advantages = (b_advantages[mb_inds] - b_advantages[mb_inds].mean()) / (b_advantages[mb_inds].std() + 1e-8)
   343:                 else:
   344:                     mb_advantages = b_advantages[mb_inds]
   345: 
   346:                 loss, pg_loss, v_loss, entropy_loss, approx_kl, clipfrac = compute_losses(
   347:                     agent, b_obs[mb_inds], b_actions[mb_inds], b_logprobs[mb_inds],
   348:                     mb_advantages, b_returns[mb_inds], b_values[mb_inds], args,
   349:                 )
   350:                 clipfracs.append(clipfrac)
   351: 
   352:                 optimizer.zero_grad()
   353:                 loss.backward()
   354:                 nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
   355:                 optimizer.step()
   356: 
   357:             if args.target_kl is not None and approx_kl > args.target_kl:
   358:                 break
   359: 
   360:         # Log training metrics
   361:         print(
   362:             f"TRAIN_METRICS step={global_step} "
   363:             f"pg_loss={pg_loss.item():.4f} vf_loss={v_loss.item():.4f} "
   364:             f"entropy={entropy_loss.item():.4f} approx_kl={approx_kl.item():.4f} "
   365:             f"clipfrac={np.mean(clipfracs):.4f}",
   366:             flush=True,
   367:         )
   368: 
   369:         # Evaluation
   370:         if global_step % args.eval_freq < args.batch_size:
   371:             _env = envs.envs[0]
   372:             _obs_rms = None
   373:             while hasattr(_env, 'env'):
   374:                 if isinstance(_env, gym.wrappers.NormalizeObservation):
   375:                     _obs_rms = _env.obs_rms
   376:                     break
   377:                 _env = _env.env
   378:             eval_returns = eval_agent(
   379:                 args.env_id, agent, device,
   380:                 n_episodes=args.eval_episodes, seed=args.seed + 1000,
   381:                 gamma=args.gamma, obs_rms=_obs_rms,
   382:             )
   383:             mean_return = eval_returns.mean()
   384:             print(f"Eval episodic_return: {mean_return:.2f}", flush=True)
   385: 
   386:     envs.close()
```




## How You Will Be Evaluated

After you finish, evaluation runs a fixed set of scripts and aggregates the
metrics they emit. These scripts are **not** in your workspace — you cannot
read or modify them. The labels below indicate what each evaluation tests:

- **halfcheetah-v4** — wall-clock budget `05:00:00`, compute share `0.33`
- **swimmer-v4** — wall-clock budget `05:00:00`, compute share `0.33`
- **inverteddoublependulum-v4** — wall-clock budget `04:00:00`, compute share `0.25`


Scoring uses the same `combined_score` aggregation as the MLS-Bench
leaderboard. Multiple seeds are averaged.



## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `ppo` baseline — editable region  [READ-ONLY — reference implementation]

In `cleanrl/cleanrl/custom_onpolicy_continuous.py`:

```python
Lines 175–218:
   172:     # =================================================================
   173:     # EDITABLE: get_action_and_value and compute_losses
   174:     # =================================================================
   175:     def get_action_and_value(self, obs, action=None):
   176:         action_mean = self.actor_mean(obs)
   177:         action_logstd = self.actor_logstd.expand_as(action_mean)
   178:         action_std = torch.exp(action_logstd)
   179:         probs = Normal(action_mean, action_std)
   180:         if action is None:
   181:             action = probs.sample()
   182:         return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)
   183: 
   184: 
   185: def compute_losses(agent, mb_obs, mb_actions, mb_logprobs, mb_advantages, mb_returns, mb_values, args):
   186:     """PPO clipped surrogate objective + clipped value loss."""
   187:     _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions)
   188:     logratio = newlogprob - mb_logprobs
   189:     ratio = logratio.exp()
   190: 
   191:     with torch.no_grad():
   192:         approx_kl = ((ratio - 1) - logratio).mean()
   193:         clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
   194: 
   195:     # Policy loss — clipped surrogate
   196:     pg_loss1 = -mb_advantages * ratio
   197:     pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
   198:     pg_loss = torch.max(pg_loss1, pg_loss2).mean()
   199: 
   200:     # Value loss — clipped
   201:     newvalue = newvalue.view(-1)
   202:     if args.clip_vloss:
   203:         v_loss_unclipped = (newvalue - mb_returns) ** 2
   204:         v_clipped = mb_values + torch.clamp(
   205:             newvalue - mb_values,
   206:             -args.clip_coef,
   207:             args.clip_coef,
   208:         )
   209:         v_loss_clipped = (v_clipped - mb_returns) ** 2
   210:         v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
   211:         v_loss = 0.5 * v_loss_max.mean()
   212:     else:
   213:         v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
   214: 
   215:     entropy_loss = entropy.mean()
   216:     loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
   217: 
   218:     return loss, pg_loss, v_loss, entropy_loss, approx_kl, clipfrac
   219: # =====================================================================
   220: # FIXED: Training loop
   221: # =====================================================================
```

### `awr` baseline — editable region  [READ-ONLY — reference implementation]

In `cleanrl/cleanrl/custom_onpolicy_continuous.py`:

```python
Lines 175–214:
   172:     # =================================================================
   173:     # EDITABLE: get_action_and_value and compute_losses
   174:     # =================================================================
   175:     def get_action_and_value(self, obs, action=None):
   176:         action_mean = self.actor_mean(obs)
   177:         action_logstd = self.actor_logstd.expand_as(action_mean)
   178:         action_std = torch.exp(action_logstd)
   179:         probs = Normal(action_mean, action_std)
   180:         if action is None:
   181:             action = probs.sample()
   182:         return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)
   183: 
   184: 
   185: def compute_losses(agent, mb_obs, mb_actions, mb_logprobs, mb_advantages, mb_returns, mb_values, args):
   186:     """AWR: advantage-weighted regression loss."""
   187:     _awr_beta = 0.05
   188:     _awr_max_weight = 20.0
   189: 
   190:     _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions)
   191:     logratio = newlogprob - mb_logprobs
   192:     ratio = logratio.exp()
   193: 
   194:     with torch.no_grad():
   195:         approx_kl = ((ratio - 1) - logratio).mean()
   196:         clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
   197: 
   198:     # Compute advantage weights: exp(advantage / beta), clamped for stability
   199:     with torch.no_grad():
   200:         weights = torch.exp(mb_advantages / _awr_beta)
   201:         weights = torch.clamp(weights, max=_awr_max_weight)
   202:         weights = weights / (weights.sum() + 1e-8) * weights.numel()
   203: 
   204:     # Policy loss — advantage-weighted regression (supervised)
   205:     pg_loss = -(newlogprob * weights).mean()
   206: 
   207:     # Value loss — simple MSE
   208:     newvalue = newvalue.view(-1)
   209:     v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
   210: 
   211:     entropy_loss = entropy.mean()
   212:     loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
   213: 
   214:     return loss, pg_loss, v_loss, entropy_loss, approx_kl, clipfrac
   215: # =====================================================================
   216: # FIXED: Training loop
   217: # =====================================================================
```

### `ppo_penalty` baseline — editable region  [READ-ONLY — reference implementation]

In `cleanrl/cleanrl/custom_onpolicy_continuous.py`:

```python
Lines 175–219:
   172:     # =================================================================
   173:     # EDITABLE: get_action_and_value and compute_losses
   174:     # =================================================================
   175:     def get_action_and_value(self, obs, action=None):
   176:         action_mean = self.actor_mean(obs)
   177:         action_logstd = self.actor_logstd.expand_as(action_mean)
   178:         action_std = torch.exp(action_logstd)
   179:         probs = Normal(action_mean, action_std)
   180:         if action is None:
   181:             action = probs.sample()
   182:         return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)
   183: 
   184: 
   185: def compute_losses(agent, mb_obs, mb_actions, mb_logprobs, mb_advantages, mb_returns, mb_values, args):
   186:     """PPO-Penalty: adaptive KL penalty instead of clipped surrogate."""
   187:     if not hasattr(agent, '_kl_beta'):
   188:         agent._kl_beta = 0.5
   189:         agent._target_kl = 0.01
   190: 
   191:     _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions)
   192:     logratio = newlogprob - mb_logprobs
   193:     ratio = logratio.exp()
   194: 
   195:     # KL divergence — WITH gradient for the penalty term
   196:     kl = ((ratio - 1) - logratio).mean()
   197: 
   198:     with torch.no_grad():
   199:         approx_kl = kl.detach()
   200:         clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
   201: 
   202:     # Policy loss — KL-penalized (no clipping)
   203:     pg_loss = -(mb_advantages * ratio).mean() + agent._kl_beta * kl
   204: 
   205:     # Adapt KL penalty coefficient
   206:     with torch.no_grad():
   207:         if approx_kl > 1.5 * agent._target_kl:
   208:             agent._kl_beta = min(agent._kl_beta * 2.0, 100.0)
   209:         elif approx_kl < agent._target_kl / 1.5:
   210:             agent._kl_beta = max(agent._kl_beta / 2.0, 1e-4)
   211: 
   212:     # Value loss — simple MSE (no clipping)
   213:     newvalue = newvalue.view(-1)
   214:     v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
   215: 
   216:     entropy_loss = entropy.mean()
   217:     loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
   218: 
   219:     return loss, pg_loss, v_loss, entropy_loss, approx_kl, clipfrac
   220: # =====================================================================
   221: # FIXED: Training loop
   222: # =====================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
