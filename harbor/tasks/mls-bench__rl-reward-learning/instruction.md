# MLS-Bench: rl-reward-learning

# Inverse RL: Reward Learning from Expert Demonstrations

## Research Question
Design and implement an inverse reinforcement learning (IRL) algorithm
that learns a reward function from expert demonstrations. Your code goes
in `custom_irl.py`, specifically the `RewardNetwork` and `IRLAlgorithm`
classes. Several reference implementations from the `imitation` library
are provided as read-only `*.edit.py` baselines.

## Background
Inverse reinforcement learning recovers a reward function that explains
observed expert behavior. The learned reward is then used to train a
policy via standard RL — PPO in this benchmark — and the resulting
policy is scored against the true environment reward. Key challenges
include:

- Designing reward network architectures that capture the structure of
  expert behavior.
- Balancing discriminator / reward training with policy improvement.
- Avoiding reward hacking, where the policy exploits artifacts in the
  learned reward.
- Ensuring the learned reward generalizes across the state distribution
  visited during policy training.

Reference baselines spanning the design space:
- **BC** — supervised behavior cloning on the expert state-action
  pairs. Does not learn a reward.
- **GAIL** — Ho and Ermon, "Generative Adversarial Imitation Learning"
  (arXiv:1606.03476, NeurIPS 2016). Adversarial discriminator between
  expert and policy occupancy; the policy is trained with the
  discriminator-derived reward.
- **AIRL** — Fu et al., "Learning Robust Rewards with Adversarial
  Inverse Reinforcement Learning" (arXiv:1710.11248, ICLR 2018).
  Adversarial reward learning with a state-only reward decomposition
  that yields a reward robust to dynamics changes.

## Evaluation
Trained and evaluated on Gymnasium MuJoCo locomotion environments
including HalfCheetah-v4, Hopper-v4 and Walker2d-v4 using
pre-generated expert demonstrations bundled with the benchmark. The
PPO policy is trained with the learned reward signal and evaluated
under the true environment reward. Metric: mean episodic return over
evaluation episodes (higher is better). Strong methods should learn
rewards that generalize across state distributions and task dynamics.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/imitation/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to score zero.

- `imitation/custom_irl.py`
- editable lines **231–357**


Other files you may **read** for context (do not modify):
- `imitation/src/imitation/rewards/reward_nets.py`


## Readable Context


### `imitation/custom_irl.py`  [EDITABLE — lines 231–357 only]

```python
     1: # Custom IRL / Reward Learning algorithm for MLS-Bench
     2: #
     3: # EDITABLE section: RewardNetwork and IRLAlgorithm classes.
     4: # FIXED sections: everything else (config, env, demo loading, PPO training, evaluation).
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
    16: 
    17: 
    18: # =====================================================================
    19: # FIXED: Configuration
    20: # =====================================================================
    21: @dataclass
    22: class Args:
    23:     env_id: str = "HalfCheetah-v4"
    24:     seed: int = 42
    25:     torch_deterministic: bool = True
    26:     cuda: bool = True
    27:     # IRL training
    28:     irl_epochs: int = 200
    29:     irl_batch_size: int = 256
    30:     irl_lr: float = 3e-4
    31:     demo_path: str = ""  # set from env or CLI
    32:     # Policy training (PPO via custom loop)
    33:     total_timesteps: int = 1000000
    34:     policy_lr: float = 3e-4
    35:     gamma: float = 0.99
    36:     gae_lambda: float = 0.95
    37:     n_steps: int = 2048
    38:     n_epochs: int = 10
    39:     minibatch_size: int = 64
    40:     clip_coef: float = 0.2
    41:     ent_coef: float = 0.0
    42:     vf_coef: float = 0.5
    43:     max_grad_norm: float = 0.5
    44:     # Evaluation
    45:     eval_freq: int = 50000
    46:     eval_episodes: int = 10
    47:     # IRL-specific
    48:     n_gen_steps_per_irl_update: int = 2048
    49:     n_irl_updates_per_round: int = 5
    50: 
    51: 
    52: # =====================================================================
    53: # FIXED: Environment setup
    54: # =====================================================================
    55: def make_env(env_id, seed, idx=0):
    56:     def thunk():
    57:         env = gym.make(env_id)
    58:         env = gym.wrappers.RecordEpisodeStatistics(env)
    59:         env.action_space.seed(seed + idx)
    60:         env.observation_space.seed(seed + idx)
    61:         return env
    62:     return thunk
    63: 
    64: 
    65: # =====================================================================
    66: # FIXED: Expert demonstration generation & loading
    67: # =====================================================================
    68: def generate_expert_demos(demo_path, env_id, total_timesteps=2_000_000, n_demos=25000):
    69:     """Train PPO expert and collect demonstrations on GPU."""
    70:     from stable_baselines3 import PPO
    71:     from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    72:     from stable_baselines3.common.evaluation import evaluate_policy as sb3_eval
    73: 
    74:     os.makedirs(demo_path, exist_ok=True)
    75:     print(f"Training expert for {env_id} ({total_timesteps} steps)...", flush=True)
    76: 
    77:     train_env = SubprocVecEnv([lambda eid=env_id, i=i: gym.make(eid) for i in range(4)])
    78:     sb3_device = "cuda" if torch.cuda.is_available() else "cpu"
    79:     model = PPO("MlpPolicy", train_env, verbose=0,
    80:                 n_steps=2048, batch_size=64, n_epochs=10,
    81:                 learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
    82:                 clip_range=0.2, ent_coef=0.0, vf_coef=0.5,
    83:                 max_grad_norm=0.5, device=sb3_device)
    84:     model.learn(total_timesteps=total_timesteps)
    85:     train_env.close()
    86: 
    87:     eval_env = DummyVecEnv([lambda eid=env_id: gym.make(eid)])
    88:     mean_reward, std_reward = sb3_eval(model, eval_env, n_eval_episodes=20)
    89:     print(f"  Expert {env_id}: {mean_reward:.1f} +/- {std_reward:.1f}", flush=True)
    90:     model.save(os.path.join(demo_path, f"{env_id}_expert"))
    91: 
    92:     all_obs, all_acts, all_next_obs, all_dones = [], [], [], []
    93:     obs = eval_env.reset()
    94:     for _ in range(n_demos):
    95:         action, _ = model.predict(obs, deterministic=True)
    96:         next_obs, reward, done, info = eval_env.step(action)
    97:         all_obs.append(obs[0].copy())
    98:         all_acts.append(action[0].copy())
    99:         all_next_obs.append(next_obs[0].copy())
   100:         all_dones.append(float(done[0]))
   101:         obs = next_obs
   102: 
   103:     demos = {
   104:         "obs": np.array(all_obs, dtype=np.float32),
   105:         "acts": np.array(all_acts, dtype=np.float32),
   106:         "next_obs": np.array(all_next_obs, dtype=np.float32),
   107:         "dones": np.array(all_dones, dtype=np.float32),
   108:     }
   109:     np.savez(os.path.join(demo_path, f"{env_id}_demos.npz"), **demos)
   110:     print(f"  Saved {n_demos} transitions for {env_id}", flush=True)
   111:     eval_env.close()
   112: 
   113: 
   114: def load_expert_demos(demo_path, env_id, device):
   115:     """Load expert demonstrations, generating them if needed."""
   116:     path = os.path.join(demo_path, f"{env_id}_demos.npz")
   117:     if not os.path.exists(path):
   118:         generate_expert_demos(demo_path, env_id)
   119:     data = np.load(path)
   120:     demos = {
   121:         "obs": torch.tensor(data["obs"], dtype=torch.float32, device=device),
   122:         "acts": torch.tensor(data["acts"], dtype=torch.float32, device=device),
   123:         "next_obs": torch.tensor(data["next_obs"], dtype=torch.float32, device=device),
   124:         "dones": torch.tensor(data["dones"], dtype=torch.float32, device=device),
   125:     }
   126:     print(f"Loaded {len(demos['obs'])} expert transitions from {path}")
   127:     return demos
   128: 
   129: 
   130: # =====================================================================
   131: # FIXED: Policy network (PPO actor-critic, not editable)
   132: # =====================================================================
   133: def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
   134:     nn.init.orthogonal_(layer.weight, std)
   135:     nn.init.constant_(layer.bias, bias_const)
   136:     return layer
   137: 
   138: 
   139: class PolicyNetwork(nn.Module):
   140:     """PPO Actor-Critic policy. FIXED — not editable."""
   141: 
   142:     def __init__(self, obs_dim, action_dim):
   143:         super().__init__()
   144:         self.critic = nn.Sequential(
   145:             layer_init(nn.Linear(obs_dim, 256)),
   146:             nn.Tanh(),
   147:             layer_init(nn.Linear(256, 256)),
   148:             nn.Tanh(),
   149:             layer_init(nn.Linear(256, 1), std=1.0),
   150:         )
   151:         self.actor_mean = nn.Sequential(
   152:             layer_init(nn.Linear(obs_dim, 256)),
   153:             nn.Tanh(),
   154:             layer_init(nn.Linear(256, 256)),
   155:             nn.Tanh(),
   156:             layer_init(nn.Linear(256, action_dim), std=0.01),
   157:         )
   158:         self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
   159: 
   160:     def get_value(self, x):
   161:         return self.critic(x)
   162: 
   163:     def get_action_and_value(self, x, action=None):
   164:         action_mean = self.actor_mean(x)
   165:         action_logstd = self.actor_logstd.expand_as(action_mean)
   166:         action_std = torch.exp(action_logstd)
   167:         probs = torch.distributions.Normal(action_mean, action_std)
   168:         if action is None:
   169:             action = probs.sample()
   170:         return action, probs.log_prob(action).sum(-1), probs.entropy().sum(-1), self.critic(x)
   171: 
   172: 
   173: # =====================================================================
   174: # FIXED: Rollout buffer for PPO
   175: # =====================================================================
   176: class RolloutBuffer:
   177:     """Stores rollout data for PPO updates."""
   178: 
   179:     def __init__(self, n_steps, obs_dim, action_dim, device):
   180:         self.obs = torch.zeros((n_steps, obs_dim), device=device)
   181:         self.actions = torch.zeros((n_steps, action_dim), device=device)
   182:         self.logprobs = torch.zeros(n_steps, device=device)
   183:         self.rewards = torch.zeros(n_steps, device=device)
   184:         self.dones = torch.zeros(n_steps, device=device)
   185:         self.values = torch.zeros(n_steps, device=device)
   186:         self.next_obs = torch.zeros((n_steps, obs_dim), device=device)
   187:         self.ptr = 0
   188: 
   189:     def add(self, obs, action, logprob, reward, done, value, next_obs):
   190:         self.obs[self.ptr] = obs
   191:         self.actions[self.ptr] = action
   192:         self.logprobs[self.ptr] = logprob
   193:         self.rewards[self.ptr] = reward
   194:         self.dones[self.ptr] = done
   195:         self.values[self.ptr] = value
   196:         self.next_obs[self.ptr] = next_obs
   197:         self.ptr += 1
   198: 
   199:     def reset(self):
   200:         self.ptr = 0
   201: 
   202: 
   203: # =====================================================================
   204: # FIXED: Evaluation
   205: # =====================================================================
   206: @torch.no_grad()
   207: def evaluate_policy(env_id, policy, device, n_episodes, seed):
   208:     """Evaluate policy for n_episodes; returns array of episode rewards."""
   209:     eval_env = gym.make(env_id)
   210:     episode_rewards = []
   211:     for ep in range(n_episodes):
   212:         obs, _ = eval_env.reset(seed=seed + ep)
   213:         done = False
   214:         episode_reward = 0.0
   215:         while not done:
   216:             obs_t = torch.tensor(obs.reshape(1, -1), device=device, dtype=torch.float32)
   217:             action, _, _, _ = policy.get_action_and_value(obs_t)
   218:             action = action.cpu().numpy().flatten()
   219:             action = np.clip(action, eval_env.action_space.low, eval_env.action_space.high)
   220:             obs, reward, terminated, truncated, _ = eval_env.step(action)
   221:             done = terminated or truncated
   222:             episode_reward += reward
   223:         episode_rewards.append(episode_reward)
   224:     eval_env.close()
   225:     return np.asarray(episode_rewards)
   226: 
   227: 
   228: # =====================================================================
   229: # EDITABLE: Reward Network and IRL Algorithm
   230: # =====================================================================
   231: class RewardNetwork(nn.Module):
   232:     """Reward network R(s, a, s') -> scalar.
   233: 
   234:     Takes state, action, next_state as input and outputs a scalar reward.
   235:     This is the discriminator/reward model used in IRL.
   236: 
   237:     You may redesign this architecture entirely. The forward signature must remain:
   238:         forward(state, action, next_state) -> reward_tensor of shape (batch,)
   239:     """
   240: 
   241:     def __init__(self, obs_dim, action_dim):
   242:         super().__init__()
   243:         input_dim = obs_dim + action_dim + obs_dim
   244:         self.net = nn.Sequential(
   245:             nn.Linear(input_dim, 256),
   246:             nn.ReLU(),
   247:             nn.Linear(256, 256),
   248:             nn.ReLU(),
   249:             nn.Linear(256, 1),
   250:         )
   251: 
   252:     def forward(self, state, action, next_state):
   253:         """Compute reward for a batch of transitions.
   254: 
   255:         Args:
   256:             state: (batch, obs_dim) current observations
   257:             action: (batch, action_dim) actions taken
   258:             next_state: (batch, obs_dim) next observations
   259: 
   260:         Returns:
   261:             Reward tensor of shape (batch,)
   262:         """
   263:         x = torch.cat([state, action, next_state], dim=-1)
   264:         return self.net(x).squeeze(-1)
   265: 
   266: 
   267: class IRLAlgorithm:
   268:     """Inverse RL / Reward Learning algorithm.
   269: 
   270:     Responsible for:
   271:       1. Training the reward network to distinguish expert from policy data.
   272:       2. Providing learned rewards for policy training.
   273: 
   274:     The main training loop calls:
   275:         irl = IRLAlgorithm(reward_net, expert_demos, obs_dim, action_dim, device, args)
   276:         ...
   277:         # After collecting on-policy rollout data:
   278:         irl.update(policy_obs, policy_acts, policy_next_obs, policy_dones)
   279:         ...
   280:         # To compute rewards for PPO:
   281:         rewards = irl.compute_reward(obs, acts, next_obs)
   282: 
   283:     Available classes (defined above, editable):
   284:         RewardNetwork — R(s, a, s') -> scalar
   285: 
   286:     You MUST keep:
   287:         - self.reward_net set to a RewardNetwork instance
   288:         - compute_reward(obs, acts, next_obs) -> tensor of shape (batch,)
   289:         - update(...) that trains the reward network
   290:     """
   291: 
   292:     def __init__(self, reward_net, expert_demos, obs_dim, action_dim, device, args):
   293:         self.reward_net = reward_net
   294:         self.expert_demos = expert_demos
   295:         self.device = device
   296:         self.args = args
   297:         self.obs_dim = obs_dim
   298:         self.action_dim = action_dim
   299: 
   300:         self.optimizer = optim.Adam(self.reward_net.parameters(), lr=args.irl_lr)
   301:         self.total_updates = 0
   302: 
   303:     def compute_reward(self, obs, acts, next_obs):
   304:         """Compute learned reward for given transitions (used during PPO rollout).
   305: 
   306:         Args:
   307:             obs: (batch, obs_dim) observations
   308:             acts: (batch, action_dim) actions
   309:             next_obs: (batch, obs_dim) next observations
   310: 
   311:         Returns:
   312:             Reward tensor of shape (batch,)
   313:         """
   314:         with torch.no_grad():
   315:             return self.reward_net(obs, acts, next_obs)
   316: 
   317:     def update(self, policy_obs, policy_acts, policy_next_obs, policy_dones):
   318:         """Update reward network using expert demos and on-policy generator data.
   319: 
   320:         Args:
   321:             policy_obs: (N, obs_dim) observations from current policy rollout
   322:             policy_acts: (N, action_dim) actions from current policy rollout
   323:             policy_next_obs: (N, obs_dim) next observations from policy rollout
   324:             policy_dones: (N,) done flags from policy rollout
   325: 
   326:         Returns:
   327:             dict of scalar metrics for logging
   328: 
   329:         TODO: Implement your IRL reward learning algorithm here.
   330:         """
   331:         self.total_updates += 1
   332:         batch_size = self.args.irl_batch_size
   333: 
   334:         # Sample expert data
   335:         n_expert = len(self.expert_demos["obs"])
   336:         expert_idx = torch.randint(0, n_expert, (batch_size,))
   337:         expert_obs = self.expert_demos["obs"][expert_idx]
   338:         expert_acts = self.expert_demos["acts"][expert_idx]
   339:         expert_next_obs = self.expert_demos["next_obs"][expert_idx]
   340: 
   341:         # Sample policy data
   342:         n_policy = len(policy_obs)
   343:         policy_idx = torch.randint(0, n_policy, (batch_size,))
   344:         gen_obs = policy_obs[policy_idx]
   345:         gen_acts = policy_acts[policy_idx]
   346:         gen_next_obs = policy_next_obs[policy_idx]
   347: 
   348:         # Placeholder — replace with your IRL algorithm
   349:         loss = torch.tensor(0.0, device=self.device)
   350: 
   351:         self.optimizer.zero_grad()
   352:         loss.backward()
   353:         self.optimizer.step()
   354: 
   355:         return {"irl_loss": loss.item()}
   356: 
   357: 
   358: # =====================================================================
   359: # FIXED: PPO update step
   360: # =====================================================================
   361: def ppo_update(policy, optimizer, buffer, args, device):
   362:     """Run PPO update on the rollout buffer. Returns metrics dict."""
   363:     n_steps = buffer.ptr
   364:     obs = buffer.obs[:n_steps]
   365:     actions = buffer.actions[:n_steps]
   366:     logprobs = buffer.logprobs[:n_steps]
   367:     rewards = buffer.rewards[:n_steps]
   368:     dones = buffer.dones[:n_steps]
   369:     values = buffer.values[:n_steps]
   370: 
   371:     # Compute GAE
   372:     with torch.no_grad():
   373:         next_value = policy.get_value(buffer.next_obs[n_steps - 1].unsqueeze(0)).squeeze()
   374:     advantages = torch.zeros(n_steps, device=device)
   375:     lastgaelam = 0
   376:     for t in reversed(range(n_steps)):
   377:         if t == n_steps - 1:
   378:             nextnonterminal = 1.0 - dones[t]
   379:             nextvalue = next_value
   380:         else:
   381:             nextnonterminal = 1.0 - dones[t]
   382:             nextvalue = values[t + 1]
   383:         delta = rewards[t] + args.gamma * nextvalue * nextnonterminal - values[t]
   384:         advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
   385:     returns = advantages + values
   386: 
   387:     # PPO epochs
   388:     indices = np.arange(n_steps)
   389:     total_pg_loss = 0.0
   390:     total_v_loss = 0.0
   391:     total_entropy = 0.0
   392:     n_updates = 0
   393: 
   394:     for epoch in range(args.n_epochs):
   395:         np.random.shuffle(indices)
   396:         for start in range(0, n_steps, args.minibatch_size):
   397:             end = start + args.minibatch_size
   398:             if end > n_steps:
   399:                 break
   400:             mb_idx = indices[start:end]
   401: 
   402:             _, newlogprob, entropy, newvalue = policy.get_action_and_value(
   403:                 obs[mb_idx], actions[mb_idx]
   404:             )
   405:             logratio = newlogprob - logprobs[mb_idx]
   406:             ratio = logratio.exp()
   407: 
   408:             mb_advantages = advantages[mb_idx]
   409:             mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
   410: 
   411:             # Policy loss
   412:             pg_loss1 = -mb_advantages * ratio
   413:             pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
   414:             pg_loss = torch.max(pg_loss1, pg_loss2).mean()
   415: 
   416:             # Value loss
   417:             v_loss = F.mse_loss(newvalue.squeeze(), returns[mb_idx])
   418: 
   419:             # Entropy loss
   420:             entropy_loss = entropy.mean()
   421: 
   422:             loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss
   423: 
   424:             optimizer.zero_grad()
   425:             loss.backward()
   426:             nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
   427:             optimizer.step()
   428: 
   429:             total_pg_loss += pg_loss.item()
   430:             total_v_loss += v_loss.item()
   431:             total_entropy += entropy_loss.item()
   432:             n_updates += 1
   433: 
   434:     return {
   435:         "pg_loss": total_pg_loss / max(n_updates, 1),
   436:         "v_loss": total_v_loss / max(n_updates, 1),
   437:         "entropy": total_entropy / max(n_updates, 1),
   438:     }
   439: 
   440: 
   441: # =====================================================================
   442: # FIXED: Main training loop
   443: # =====================================================================
   444: if __name__ == "__main__":
   445:     import argparse
   446: 
   447:     parser = argparse.ArgumentParser()
   448:     parser.add_argument("--env-id", type=str, default="HalfCheetah-v4")
   449:     parser.add_argument("--seed", type=int, default=42)
   450:     parser.add_argument("--total-timesteps", type=int, default=1000000)
   451:     parser.add_argument("--demo-path", type=str, default="")
   452:     cli_args = parser.parse_args()
   453: 
   454:     args = Args()
   455:     args.env_id = cli_args.env_id
   456:     args.seed = cli_args.seed
   457:     args.total_timesteps = cli_args.total_timesteps
   458:     # Demo path: CLI > env SAVE_PATH > fallback
   459:     args.demo_path = cli_args.demo_path or os.path.join(
   460:         os.environ.get("SAVE_PATH", "/workspace"), "irl_experts"
   461:     )
   462: 
   463:     # Seeding
   464:     random.seed(args.seed)
   465:     np.random.seed(args.seed)
   466:     torch.manual_seed(args.seed)
   467:     torch.backends.cudnn.deterministic = args.torch_deterministic
   468: 
   469:     device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
   470: 
   471:     # Environment setup
   472:     env = gym.make(args.env_id)
   473:     env = gym.wrappers.RecordEpisodeStatistics(env)
   474:     env.action_space.seed(args.seed)
   475: 
   476:     obs_dim = int(np.prod(env.observation_space.shape))
   477:     action_dim = int(np.prod(env.action_space.shape))
   478: 
   479:     # Load expert demonstrations
   480:     expert_demos = load_expert_demos(args.demo_path, args.env_id, device)
   481: 
   482:     # Initialize reward network and IRL algorithm
   483:     reward_net = RewardNetwork(obs_dim, action_dim).to(device)
   484:     irl = IRLAlgorithm(reward_net, expert_demos, obs_dim, action_dim, device, args)
   485: 
   486:     # ── FIXED: Parameter count check ────────────────────────────────
   487:     # Budget based on 1.05x largest baseline (AIRL with g_net + h_net).
   488:     # AIRL: g_net(obs+act -> 256 -> 256 -> 1) + h_net(obs -> 256 -> 256 -> 1)
   489:     _g_params = (obs_dim + action_dim) * 256 + 256 + 256 * 256 + 256 + 256 + 1
   490:     _h_params = obs_dim * 256 + 256 + 256 * 256 + 256 + 256 + 1
   491:     _budget = int((_g_params + _h_params + 100) * 1.05)
   492:     _total_params = sum(p.numel() for p in reward_net.parameters())
   493:     print(f"Total reward net params: {_total_params:,} (budget: {_budget:,})")
   494: 
   495:     # Initialize policy
   496:     policy = PolicyNetwork(obs_dim, action_dim).to(device)
   497:     policy_optimizer = optim.Adam(policy.parameters(), lr=args.policy_lr)
   498: 
   499:     # Allow IRL algorithm to access policy (used by BC baseline)
   500:     if hasattr(irl, "set_policy"):

[truncated: showing at most 500 lines / 60000 bytes from imitation/custom_irl.py]
```




## How You Will Be Evaluated

After you finish, evaluation runs a fixed set of scripts and aggregates the
metrics they emit. These scripts are **not** in your workspace — you cannot
read or modify them. The labels below indicate what each evaluation tests:

- **halfcheetah-v4** — wall-clock budget `12:00:00`, compute share `0.33`
- **hopper-v4** — wall-clock budget `12:00:00`, compute share `0.33`
- **walker2d-v4** — wall-clock budget `12:00:00`, compute share `0.33`


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


### `gail` baseline — editable region  [READ-ONLY — reference implementation]

In `imitation/custom_irl.py`:

```python
Lines 231–367:
   228: # =====================================================================
   229: # EDITABLE: Reward Network and IRL Algorithm
   230: # =====================================================================
   231: class _RunningMeanStd:
   232:     """Welford running mean/std for input normalization."""
   233: 
   234:     def __init__(self, shape, device, eps=1e-4):
   235:         self.mean = torch.zeros(shape, device=device, dtype=torch.float32)
   236:         self.var = torch.ones(shape, device=device, dtype=torch.float32)
   237:         self.count = eps
   238: 
   239:     @torch.no_grad()
   240:     def update(self, x):
   241:         if x.numel() == 0:
   242:             return
   243:         x = x.detach().to(self.mean.device, dtype=torch.float32).reshape(-1, self.mean.shape[-1])
   244:         batch_count = x.shape[0]
   245:         batch_mean = x.mean(0)
   246:         batch_var = x.var(0, unbiased=False) if batch_count > 1 else torch.zeros_like(self.var)
   247:         delta = batch_mean - self.mean
   248:         tot_count = self.count + batch_count
   249:         self.mean = self.mean + delta * (batch_count / tot_count)
   250:         m_a = self.var * self.count
   251:         m_b = batch_var * batch_count
   252:         M2 = m_a + m_b + (delta ** 2) * (self.count * batch_count / tot_count)
   253:         self.var = M2 / tot_count
   254:         self.count = tot_count
   255: 
   256:     def normalize(self, x):
   257:         return (x - self.mean) / torch.sqrt(self.var + 1e-8)
   258: 
   259: 
   260: class RewardNetwork(nn.Module):
   261:     """GAIL discriminator over (s,a,s'). Inputs normalized by running stats."""
   262: 
   263:     def __init__(self, obs_dim, action_dim):
   264:         super().__init__()
   265:         self.obs_dim = obs_dim
   266:         self.action_dim = action_dim
   267:         input_dim = obs_dim + action_dim + obs_dim
   268:         self.net = nn.Sequential(
   269:             nn.Linear(input_dim, 256),
   270:             nn.ReLU(),
   271:             nn.Linear(256, 256),
   272:             nn.ReLU(),
   273:             nn.Linear(256, 1),
   274:         )
   275:         self._obs_rms = None
   276:         self._update_norm = True
   277: 
   278:     def _ensure_rms(self, ref_tensor):
   279:         if self._obs_rms is None:
   280:             self._obs_rms = _RunningMeanStd((self.obs_dim,), ref_tensor.device)
   281: 
   282:     def update_obs_norm(self, obs, next_obs):
   283:         if self._obs_rms is None:
   284:             self._ensure_rms(obs)
   285:         if self._update_norm:
   286:             self._obs_rms.update(obs)
   287:             self._obs_rms.update(next_obs)
   288: 
   289:     def _norm_obs(self, x):
   290:         if self._obs_rms is None:
   291:             return x
   292:         return self._obs_rms.normalize(x)
   293: 
   294:     def forward(self, state, action, next_state):
   295:         x = torch.cat([self._norm_obs(state), action, self._norm_obs(next_state)], dim=-1)
   296:         return self.net(x).squeeze(-1)
   297: 
   298: 
   299: class IRLAlgorithm:
   300:     """GAIL — Generative Adversarial Imitation Learning.
   301: 
   302:     Discriminator D(s,a,s') is trained to output high logits for expert
   303:     transitions, low for policy. Policy reward is the imitation-library
   304:     standard transform: -log(1 - D) = -logsigmoid(-logit).
   305:     """
   306: 
   307:     def __init__(self, reward_net, expert_demos, obs_dim, action_dim, device, args):
   308:         self.reward_net = reward_net
   309:         self.expert_demos = expert_demos
   310:         self.device = device
   311:         self.args = args
   312:         self.obs_dim = obs_dim
   313:         self.action_dim = action_dim
   314: 
   315:         self.optimizer = optim.Adam(self.reward_net.parameters(), lr=args.irl_lr)
   316:         self.total_updates = 0
   317: 
   318:         # Bump effective disc updates / batch (args are FIXED).
   319:         self._inner_updates = 4
   320:         self._batch_mult = 4
   321: 
   322:     def compute_reward(self, obs, acts, next_obs):
   323:         with torch.no_grad():
   324:             logits = self.reward_net(obs, acts, next_obs)
   325:         return -F.logsigmoid(-logits)
   326: 
   327:     def update(self, policy_obs, policy_acts, policy_next_obs, policy_dones):
   328:         self.total_updates += 1
   329:         bs = self.args.irl_batch_size * self._batch_mult
   330: 
   331:         self.reward_net.update_obs_norm(policy_obs, policy_next_obs)
   332: 
   333:         n_expert = len(self.expert_demos["obs"])
   334:         n_policy = len(policy_obs)
   335: 
   336:         last = {}
   337:         for _ in range(self._inner_updates):
   338:             expert_idx = torch.randint(0, n_expert, (bs,))
   339:             expert_obs = self.expert_demos["obs"][expert_idx]
   340:             expert_acts = self.expert_demos["acts"][expert_idx]
   341:             expert_next_obs = self.expert_demos["next_obs"][expert_idx]
   342: 
   343:             policy_idx = torch.randint(0, n_policy, (bs,))
   344:             gen_obs = policy_obs[policy_idx]
   345:             gen_acts = policy_acts[policy_idx]
   346:             gen_next_obs = policy_next_obs[policy_idx]
   347: 
   348:             expert_logits = self.reward_net(expert_obs, expert_acts, expert_next_obs)
   349:             gen_logits = self.reward_net(gen_obs, gen_acts, gen_next_obs)
   350: 
   351:             logits = torch.cat([expert_logits, gen_logits], dim=0)
   352:             labels = torch.cat([
   353:                 torch.ones(bs, device=self.device),
   354:                 torch.zeros(bs, device=self.device),
   355:             ], dim=0)
   356: 
   357:             loss = F.binary_cross_entropy_with_logits(logits, labels)
   358: 
   359:             self.optimizer.zero_grad()
   360:             loss.backward()
   361:             self.optimizer.step()
   362: 
   363:             with torch.no_grad():
   364:                 acc = ((logits > 0).float() == labels).float().mean().item()
   365:             last = {"irl_loss": loss.item(), "disc_acc": acc}
   366: 
   367:         return last
   368: # =====================================================================
   369: # FIXED: PPO update step
   370: # =====================================================================
```

### `airl` baseline — editable region  [READ-ONLY — reference implementation]

In `imitation/custom_irl.py`:

```python
Lines 231–448:
   228: # =====================================================================
   229: # EDITABLE: Reward Network and IRL Algorithm
   230: # =====================================================================
   231: class _RunningMeanStd:
   232:     """Welford running mean/std for input/output normalization."""
   233: 
   234:     def __init__(self, shape, device, eps=1e-4):
   235:         self.mean = torch.zeros(shape, device=device, dtype=torch.float32)
   236:         self.var = torch.ones(shape, device=device, dtype=torch.float32)
   237:         self.count = eps
   238: 
   239:     @torch.no_grad()
   240:     def update(self, x):
   241:         if x.numel() == 0:
   242:             return
   243:         x = x.detach().to(self.mean.device, dtype=torch.float32).reshape(-1, self.mean.shape[-1]) \
   244:             if self.mean.dim() else x.detach().to(self.mean.device, dtype=torch.float32).reshape(-1)
   245:         batch_count = x.shape[0]
   246:         batch_mean = x.mean(0)
   247:         batch_var = x.var(0, unbiased=False) if batch_count > 1 else torch.zeros_like(self.var)
   248:         delta = batch_mean - self.mean
   249:         tot_count = self.count + batch_count
   250:         self.mean = self.mean + delta * (batch_count / tot_count)
   251:         m_a = self.var * self.count
   252:         m_b = batch_var * batch_count
   253:         M2 = m_a + m_b + (delta ** 2) * (self.count * batch_count / tot_count)
   254:         self.var = M2 / tot_count
   255:         self.count = tot_count
   256: 
   257:     def normalize(self, x):
   258:         return (x - self.mean) / torch.sqrt(self.var + 1e-8)
   259: 
   260: 
   261: class RewardNetwork(nn.Module):
   262:     """AIRL shaped reward: f(s,a,s',done) = g(s,a) + gamma * (1-done) * h(s') - h(s).
   263: 
   264:     Adds RunningNorm on the obs inputs (frozen during inference) and on
   265:     the network output, mirroring imitation's tuned configs.
   266:     """
   267: 
   268:     def __init__(self, obs_dim, action_dim):
   269:         super().__init__()
   270:         self.obs_dim = obs_dim
   271:         self.action_dim = action_dim
   272: 
   273:         # g(s, a): reward approximator
   274:         self.g_net = nn.Sequential(
   275:             nn.Linear(obs_dim + action_dim, 256),
   276:             nn.ReLU(),
   277:             nn.Linear(256, 256),
   278:             nn.ReLU(),
   279:             nn.Linear(256, 1),
   280:         )
   281:         # h(s): potential-based shaping
   282:         self.h_net = nn.Sequential(
   283:             nn.Linear(obs_dim, 256),
   284:             nn.ReLU(),
   285:             nn.Linear(256, 256),
   286:             nn.ReLU(),
   287:             nn.Linear(256, 1),
   288:         )
   289:         self.gamma = 0.99
   290: 
   291:         # Running normalization. Initialized lazily (we need a device).
   292:         self._obs_rms = None
   293:         self._out_rms = None
   294:         self._update_norm = True
   295: 
   296:     def _ensure_rms(self, ref_tensor):
   297:         if self._obs_rms is None:
   298:             self._obs_rms = _RunningMeanStd((self.obs_dim,), ref_tensor.device)
   299:             self._out_rms = _RunningMeanStd((1,), ref_tensor.device)
   300: 
   301:     def freeze_norm(self):
   302:         self._update_norm = False
   303: 
   304:     def update_obs_norm(self, obs, next_obs):
   305:         if self._obs_rms is None:
   306:             self._ensure_rms(obs)
   307:         if self._update_norm:
   308:             self._obs_rms.update(obs)
   309:             self._obs_rms.update(next_obs)
   310: 
   311:     def update_out_norm(self, raw_f):
   312:         if self._out_rms is None:
   313:             return
   314:         if self._update_norm:
   315:             self._out_rms.update(raw_f.unsqueeze(-1) if raw_f.dim() == 1 else raw_f)
   316: 
   317:     def _norm_obs(self, x):
   318:         if self._obs_rms is None:
   319:             return x
   320:         return self._obs_rms.normalize(x)
   321: 
   322:     def _norm_out(self, y):
   323:         if self._out_rms is None:
   324:             return y
   325:         return (y - self._out_rms.mean.squeeze()) / torch.sqrt(self._out_rms.var.squeeze() + 1e-8)
   326: 
   327:     def g(self, state, action):
   328:         x = torch.cat([self._norm_obs(state), action], dim=-1)
   329:         return self.g_net(x).squeeze(-1)
   330: 
   331:     def h(self, state):
   332:         return self.h_net(self._norm_obs(state)).squeeze(-1)
   333: 
   334:     def raw_f(self, state, action, next_state, done=None):
   335:         """Unnormalized shaped reward. done: float tensor of shape (batch,)."""
   336:         h_next = self.h(next_state)
   337:         if done is not None:
   338:             h_next = h_next * (1.0 - done.float())
   339:         return self.g(state, action) + self.gamma * h_next - self.h(state)
   340: 
   341:     def forward(self, state, action, next_state, done=None):
   342:         """Shaped reward, with running output normalization applied."""
   343:         f = self.raw_f(state, action, next_state, done)
   344:         return self._norm_out(f)
   345: 
   346: 
   347: class IRLAlgorithm:
   348:     """AIRL — Adversarial Inverse Reinforcement Learning.
   349: 
   350:     Discriminator logits are raw_f(s,a,s',done) - log pi(a|s).
   351:     The reward returned to PPO is the *normalized* shaped f, so the FIXED
   352:     template-level reward normalization (running mean/std on buffer.rewards)
   353:     becomes near-identity rather than collapsing the signal.
   354:     """
   355: 
   356:     def __init__(self, reward_net, expert_demos, obs_dim, action_dim, device, args):
   357:         self.reward_net = reward_net
   358:         self.expert_demos = expert_demos
   359:         self.device = device
   360:         self.args = args
   361:         self.obs_dim = obs_dim
   362:         self.action_dim = action_dim
   363: 
   364:         self.optimizer = optim.Adam(self.reward_net.parameters(), lr=args.irl_lr)
   365:         self.total_updates = 0
   366:         self._policy = None
   367: 
   368:         # Effective per-round disc updates and batch (args are FIXED outside
   369:         # editable range — bump internally to match imitation tuned configs).
   370:         self._inner_updates = 4   # multiplies args.n_irl_updates_per_round
   371:         self._batch_mult = 4      # multiplies args.irl_batch_size
   372: 
   373:     def set_policy(self, policy, optimizer):
   374:         del optimizer
   375:         self._policy = policy
   376: 
   377:     def compute_reward(self, obs, acts, next_obs):
   378:         """Normalized shaped reward for PPO. dones not available here, so
   379:         the terminal correction is applied during update() instead."""
   380:         with torch.no_grad():
   381:             return self.reward_net(obs, acts, next_obs)
   382: 
   383:     def _log_policy_act_prob(self, obs, acts):
   384:         if self._policy is None:
   385:             raise RuntimeError("AIRL requires set_policy() before discriminator updates")
   386:         with torch.no_grad():
   387:             _, log_prob, _, _ = self._policy.get_action_and_value(obs, acts)
   388:         return log_prob.detach()
   389: 
   390:     def update(self, policy_obs, policy_acts, policy_next_obs, policy_dones):
   391:         """AIRL discriminator update using raw_f - log pi(a|s) on the BCE side."""
   392:         self.total_updates += 1
   393:         bs = self.args.irl_batch_size * self._batch_mult
   394: 
   395:         # Update obs running stats once per outer call using the freshest
   396:         # generator rollout — keeps statistics close to the policy state dist.
   397:         self.reward_net.update_obs_norm(policy_obs, policy_next_obs)
   398: 
   399:         n_expert = len(self.expert_demos["obs"])
   400:         n_policy = len(policy_obs)
   401:         # Expert "dones" are not stored in the demo dict — assume non-terminal
   402:         # (correct for halfcheetah which has no terminal; mildly wrong for
   403:         # hopper/walker terminal states but reference imitation also lacks
   404:         # dones for expert demos by default).
   405:         expert_done_zeros = torch.zeros(bs, device=self.device)
   406: 
   407:         last = {}
   408:         for _ in range(self._inner_updates):
   409:             # Resample fresh batches each inner step
   410:             expert_idx = torch.randint(0, n_expert, (bs,))
   411:             expert_obs = self.expert_demos["obs"][expert_idx]
   412:             expert_acts = self.expert_demos["acts"][expert_idx]
   413:             expert_next_obs = self.expert_demos["next_obs"][expert_idx]
   414: 
   415:             policy_idx = torch.randint(0, n_policy, (bs,))
   416:             gen_obs = policy_obs[policy_idx]
   417:             gen_acts = policy_acts[policy_idx]
   418:             gen_next_obs = policy_next_obs[policy_idx]
   419:             gen_dones = policy_dones[policy_idx].float()
   420: 
   421:             expert_f = self.reward_net.raw_f(
   422:                 expert_obs, expert_acts, expert_next_obs, expert_done_zeros
   423:             )
   424:             gen_f = self.reward_net.raw_f(
   425:                 gen_obs, gen_acts, gen_next_obs, gen_dones
   426:             )
   427: 
   428:             expert_logits = expert_f - self._log_policy_act_prob(expert_obs, expert_acts)
   429:             gen_logits = gen_f - self._log_policy_act_prob(gen_obs, gen_acts)
   430: 
   431:             logits = torch.cat([expert_logits, gen_logits], dim=0)
   432:             labels = torch.cat([
   433:                 torch.ones(bs, device=self.device),
   434:                 torch.zeros(bs, device=self.device),
   435:             ], dim=0)
   436: 
   437:             loss = F.binary_cross_entropy_with_logits(logits, labels)
   438: 
   439:             self.optimizer.zero_grad()
   440:             loss.backward()
   441:             self.optimizer.step()
   442: 
   443:             with torch.no_grad():
   444:                 self.reward_net.update_out_norm(torch.cat([expert_f, gen_f], dim=0))
   445:                 acc = ((logits > 0).float() == labels).float().mean().item()
   446:             last = {"irl_loss": loss.item(), "disc_acc": acc}
   447: 
   448:         return last
   449: # =====================================================================
   450: # FIXED: PPO update step
   451: # =====================================================================
```

### `bc` baseline — editable region  [READ-ONLY — reference implementation]

In `imitation/custom_irl.py`:

```python
Lines 231–323:
   228: # =====================================================================
   229: # EDITABLE: Reward Network and IRL Algorithm
   230: # =====================================================================
   231: class RewardNetwork(nn.Module):
   232:     """Dummy reward network for BC (not used for reward shaping).
   233: 
   234:     BC does not learn a reward; this returns a constant so the PPO
   235:     loop runs but does not meaningfully update from reward signal.
   236:     The policy is trained via supervised loss in IRLAlgorithm.update().
   237:     """
   238: 
   239:     def __init__(self, obs_dim, action_dim):
   240:         super().__init__()
   241:         # Unused parameters to keep interface consistent
   242:         self.dummy = nn.Linear(1, 1)
   243: 
   244:     def forward(self, state, action, next_state):
   245:         return torch.zeros(state.shape[0], device=state.device)
   246: 
   247: 
   248: class IRLAlgorithm:
   249:     """BC — Behavioral Cloning.
   250: 
   251:     Directly trains the policy network to mimic expert actions via
   252:     supervised MSE loss. The reward network is unused.
   253:     Policy is trained both via BC loss in update() and via PPO in the
   254:     main loop (with near-zero reward), but BC dominates learning.
   255:     """
   256: 
   257:     def __init__(self, reward_net, expert_demos, obs_dim, action_dim, device, args):
   258:         self.reward_net = reward_net
   259:         self.expert_demos = expert_demos
   260:         self.device = device
   261:         self.args = args
   262:         self.obs_dim = obs_dim
   263:         self.action_dim = action_dim
   264:         self.total_updates = 0
   265:         # BC does not need a reward optimizer; policy is trained externally
   266:         # We store a reference to policy that gets set during training
   267:         self._policy = None
   268:         self._policy_optimizer = None
   269: 
   270:     def set_policy(self, policy, optimizer):
   271:         """Set reference to policy for BC updates."""
   272:         self._policy = policy
   273:         self._policy_optimizer = optimizer
   274: 
   275:     def compute_reward(self, obs, acts, next_obs):
   276:         """BC uses constant reward (PPO loop is secondary)."""
   277:         return torch.zeros(obs.shape[0], device=self.device)
   278: 
   279:     def update(self, policy_obs, policy_acts, policy_next_obs, policy_dones):
   280:         """BC supervised update: minimize negative log-probability of expert actions.
   281: 
   282:         Uses the full policy distribution (mean + log_std) to compute log prob,
   283:         matching the reference imitation library approach. This trains both the
   284:         mean and the variance of the policy, giving better action coverage.
   285:         """
   286:         self.total_updates += 1
   287: 
   288:         if self._policy is None:
   289:             return {"irl_loss": 0.0}
   290: 
   291:         batch_size = self.args.irl_batch_size
   292: 
   293:         # Sample expert data
   294:         n_expert = len(self.expert_demos["obs"])
   295: 
   296:         total_bc_loss = 0.0
   297:         n_bc_steps = 20  # more BC gradient steps per IRL update
   298: 
   299:         for _ in range(n_bc_steps):
   300:             expert_idx = torch.randint(0, n_expert, (batch_size,))
   301:             expert_obs = self.expert_demos["obs"][expert_idx]
   302:             expert_acts = self.expert_demos["acts"][expert_idx]
   303: 
   304:             # Use get_action_and_value to get log_prob of expert actions
   305:             # This trains both actor_mean and actor_logstd
   306:             _, log_prob, entropy, _ = self._policy.get_action_and_value(
   307:                 expert_obs, expert_acts,
   308:             )
   309: 
   310:             # Negative log-likelihood loss (matching reference BC)
   311:             neglogp = -log_prob.mean()
   312:             # Entropy bonus for exploration (prevents policy from collapsing)
   313:             ent_bonus = -0.001 * entropy.mean()
   314: 
   315:             bc_loss = neglogp + ent_bonus
   316: 
   317:             self._policy_optimizer.zero_grad()
   318:             bc_loss.backward()
   319:             nn.utils.clip_grad_norm_(self._policy.parameters(), 0.5)
   320:             self._policy_optimizer.step()
   321:             total_bc_loss += bc_loss.item()
   322: 
   323:         return {"irl_loss": total_bc_loss / n_bc_steps}
   324: # =====================================================================
   325: # FIXED: PPO update step
   326: # =====================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
