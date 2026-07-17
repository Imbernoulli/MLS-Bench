# MLS-Bench: rl-offpolicy-sample-efficiency

# Off-Policy RL Sample Efficiency: Algorithm Design for Humanoid Locomotion

## Objective
Design a more sample-efficient off-policy reinforcement learning algorithm that achieves higher performance than FastTD3, FastSAC, and PPO on humanoid locomotion tasks within the same training budget.

## Background
FastTD3 is a high-performance off-policy RL algorithm that combines TD3 with distributional value estimation (categorical DQN with 101 atoms), parallel environments (128 envs), observation normalization, mixed-precision training, and torch.compile for speed. It uses a deterministic actor with Gaussian exploration noise and twin distributional critics with clipped double Q-learning.

Your task is to design an improved off-policy algorithm by modifying the Actor, Critic, and/or update functions. The training infrastructure (environment, replay buffer, evaluation) is fixed.

## What You Can Modify
The editable section contains:
- **Actor**: Network architecture, forward pass, exploration strategy
- **Critic**: Q-network architecture, distributional parameters, ensemble design
- **build_algorithm()**: Component construction, optimizers, schedulers, auxiliary modules
- **update_critic()**: Critic loss computation, target calculation, auxiliary objectives
- **update_actor()**: Policy gradient objective, entropy regularization, etc.
- **soft_update()**: Target network update strategy

## Key Design Dimensions
- **Architecture**: LayerNorm, spectral norm, residual connections, different activations
- **Exploration**: Noise schedule, parameter-space noise, curiosity, optimistic exploration
- **Value estimation**: Distributional RL (atoms, quantiles), ensemble methods, uncertainty
- **Policy optimization**: Entropy regularization, policy constraints, advantage weighting
- **Sample reuse**: Update-to-data ratio, replay prioritization, n-step returns
- **Representation**: Feature normalization, auxiliary losses, self-predictive representations

## Constraints
- The algorithm must work with continuous action spaces (actions clipped to [-1, 1])
- Must use the provided replay buffer and environment interface
- Total training budget: 100,000 gradient steps with 128 parallel environments
- Must produce deterministic actions at evaluation time via `actor(obs)`

## Evaluation
The algorithm is evaluated on three HumanoidBench locomotion tasks:
- **h1hand-stand-v0**: Humanoid standing balance
- **h1hand-walk-v0**: Humanoid walking
- **h1hand-run-v0**: Humanoid running

Performance is measured as mean episode return over 3 evaluation rollouts at the end of training. Higher is better.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/FastTD3/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to score zero.

- `FastTD3/fast_td3/custom_algorithm.py`
- editable lines **50–331**




## Readable Context


### `FastTD3/fast_td3/custom_algorithm.py`  [EDITABLE — lines 50–331 only]

```python
     1: """Custom off-policy RL algorithm for HumanoidBench locomotion tasks.
     2: 
     3: This script is adapted from FastTD3's training pipeline. The EDITABLE section
     4: contains the full algorithm: Actor, Critic, update functions, and exploration
     5: strategy. The FIXED sections handle environment setup, evaluation, replay buffer
     6: infrastructure, and metric printing.
     7: 
     8: The agent should design a sample-efficient off-policy (or hybrid) RL algorithm
     9: that outperforms FastTD3, FastSAC, and PPO on humanoid locomotion tasks.
    10: """
    11: 
    12: import os
    13: import sys
    14: 
    15: os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
    16: os.environ["OMP_NUM_THREADS"] = "1"
    17: if sys.platform != "darwin":
    18:     os.environ["MUJOCO_GL"] = "egl"
    19: else:
    20:     os.environ["MUJOCO_GL"] = "glfw"
    21: 
    22: import argparse
    23: import random
    24: import time
    25: import math
    26: 
    27: import tqdm
    28: import numpy as np
    29: 
    30: import torch
    31: import torch.nn as nn
    32: import torch.nn.functional as F
    33: import torch.optim as optim
    34: from torch.amp import autocast, GradScaler
    35: 
    36: from tensordict import TensorDict
    37: 
    38: # Import utilities from FastTD3
    39: sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    40: from fast_td3_utils import (
    41:     EmpiricalNormalization,
    42:     SimpleReplayBuffer,
    43:     mark_step,
    44: )
    45: 
    46: torch.set_float32_matmul_precision("high")
    47: 
    48: 
    49: # ═══════════════════════════════════════════════════════════════════════
    50: # ██ EDITABLE SECTION START — Design your off-policy RL algorithm here
    51: # ═══════════════════════════════════════════════════════════════════════
    52: 
    53: class Actor(nn.Module):
    54:     """Deterministic actor network.
    55: 
    56:     Design a better actor architecture for sample-efficient learning.
    57:     Consider: normalization layers, activation functions, initialization,
    58:     residual connections, spectral normalization, etc.
    59:     """
    60:     def __init__(self, n_obs, n_act, num_envs, device, hidden_dim=512,
    61:                  init_scale=0.01, std_min=0.001, std_max=0.4):
    62:         super().__init__()
    63:         self.n_act = n_act
    64:         self.net = nn.Sequential(
    65:             nn.Linear(n_obs, hidden_dim, device=device),
    66:             nn.ReLU(),
    67:             nn.Linear(hidden_dim, hidden_dim // 2, device=device),
    68:             nn.ReLU(),
    69:             nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
    70:             nn.ReLU(),
    71:         )
    72:         self.fc_mu = nn.Sequential(
    73:             nn.Linear(hidden_dim // 4, n_act, device=device),
    74:             nn.Tanh(),
    75:         )
    76:         nn.init.normal_(self.fc_mu[0].weight, 0.0, init_scale)
    77:         nn.init.constant_(self.fc_mu[0].bias, 0.0)
    78: 
    79:         noise_scales = (
    80:             torch.rand(num_envs, 1, device=device) * (std_max - std_min) + std_min
    81:         )
    82:         self.register_buffer("noise_scales", noise_scales)
    83:         self.register_buffer("std_min", torch.as_tensor(std_min, device=device))
    84:         self.register_buffer("std_max", torch.as_tensor(std_max, device=device))
    85:         self.n_envs = num_envs
    86:         self.device_ = device
    87: 
    88:     def forward(self, obs):
    89:         x = self.net(obs)
    90:         return self.fc_mu(x)
    91: 
    92:     def explore(self, obs, dones=None, deterministic=False):
    93:         if dones is not None and dones.sum() > 0:
    94:             new_scales = (
    95:                 torch.rand(self.n_envs, 1, device=obs.device)
    96:                 * (self.std_max - self.std_min) + self.std_min
    97:             )
    98:             dones_view = dones.view(-1, 1) > 0
    99:             self.noise_scales.copy_(
   100:                 torch.where(dones_view, new_scales, self.noise_scales)
   101:             )
   102:         act = self(obs)
   103:         if deterministic:
   104:             return act
   105:         noise = torch.randn_like(act) * self.noise_scales
   106:         return act + noise
   107: 
   108: 
   109: class DistributionalQNetwork(nn.Module):
   110:     """Distributional Q-network for value estimation.
   111: 
   112:     Design a better critic architecture for more accurate value estimation.
   113:     Consider: distributional RL atoms, network width/depth, normalization, etc.
   114:     """
   115:     def __init__(self, n_obs, n_act, num_atoms, v_min, v_max, hidden_dim, device=None):
   116:         super().__init__()
   117:         self.net = nn.Sequential(
   118:             nn.Linear(n_obs + n_act, hidden_dim, device=device),
   119:             nn.ReLU(),
   120:             nn.Linear(hidden_dim, hidden_dim // 2, device=device),
   121:             nn.ReLU(),
   122:             nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
   123:             nn.ReLU(),
   124:             nn.Linear(hidden_dim // 4, num_atoms, device=device),
   125:         )
   126:         self.v_min = v_min
   127:         self.v_max = v_max
   128:         self.num_atoms = num_atoms
   129: 
   130:     def forward(self, obs, actions):
   131:         x = torch.cat([obs, actions], 1)
   132:         return self.net(x)
   133: 
   134:     def projection(self, obs, actions, rewards, bootstrap, discount, q_support, device):
   135:         delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
   136:         batch_size = rewards.shape[0]
   137:         target_z = (
   138:             rewards.unsqueeze(1)
   139:             + bootstrap.unsqueeze(1) * discount.unsqueeze(1) * q_support
   140:         )
   141:         target_z = target_z.clamp(self.v_min, self.v_max)
   142:         b = (target_z - self.v_min) / delta_z
   143:         l = torch.floor(b).long()
   144:         u = torch.ceil(b).long()
   145:         is_int = (l == u)
   146:         l_mask = is_int & (l > 0)
   147:         u_mask = is_int & (l == 0)
   148:         l = torch.where(l_mask, l - 1, l)
   149:         u = torch.where(u_mask, u + 1, u)
   150:         next_dist = F.softmax(self.forward(obs, actions), dim=1)
   151:         proj_dist = torch.zeros_like(next_dist)
   152:         offset = (
   153:             torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size, device=device)
   154:             .unsqueeze(1).expand(batch_size, self.num_atoms).long()
   155:         )
   156:         proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
   157:         proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))
   158:         return proj_dist
   159: 
   160: 
   161: class Critic(nn.Module):
   162:     """Twin distributional critic with clipped double Q-learning.
   163: 
   164:     Design improvements to the critic: number of Q-networks, ensemble methods,
   165:     target computation strategy, etc.
   166:     """
   167:     def __init__(self, n_obs, n_act, num_atoms, v_min, v_max, hidden_dim, device=None):
   168:         super().__init__()
   169:         self.qnet1 = DistributionalQNetwork(n_obs, n_act, num_atoms, v_min, v_max, hidden_dim, device)
   170:         self.qnet2 = DistributionalQNetwork(n_obs, n_act, num_atoms, v_min, v_max, hidden_dim, device)
   171:         self.register_buffer("q_support", torch.linspace(v_min, v_max, num_atoms, device=device))
   172:         self.device = device
   173: 
   174:     def forward(self, obs, actions):
   175:         return self.qnet1(obs, actions), self.qnet2(obs, actions)
   176: 
   177:     def projection(self, obs, actions, rewards, bootstrap, discount):
   178:         q1_proj = self.qnet1.projection(obs, actions, rewards, bootstrap, discount, self.q_support, self.q_support.device)
   179:         q2_proj = self.qnet2.projection(obs, actions, rewards, bootstrap, discount, self.q_support, self.q_support.device)
   180:         return q1_proj, q2_proj
   181: 
   182:     def get_value(self, probs):
   183:         return torch.sum(probs * self.q_support, dim=1)
   184: 
   185: 
   186: def build_algorithm(n_obs, n_act, num_envs, device, args):
   187:     """Build all algorithm components: actor, critic, optimizers, schedulers.
   188: 
   189:     This function creates and returns all components needed for training.
   190:     You can modify hyperparameters, add new components (e.g., entropy tuning,
   191:     auxiliary networks, prioritized replay modifications), etc.
   192: 
   193:     Returns a dict with keys:
   194:         actor, critic, critic_target, actor_optimizer, critic_optimizer,
   195:         actor_scheduler, critic_scheduler, and any additional components.
   196:     """
   197:     actor = Actor(
   198:         n_obs=n_obs, n_act=n_act, num_envs=num_envs, device=device,
   199:         hidden_dim=args.actor_hidden_dim, init_scale=args.init_scale,
   200:         std_min=args.std_min, std_max=args.std_max,
   201:     )
   202:     critic = Critic(
   203:         n_obs=n_obs, n_act=n_act, num_atoms=args.num_atoms,
   204:         v_min=args.v_min, v_max=args.v_max,
   205:         hidden_dim=args.critic_hidden_dim, device=device,
   206:     )
   207:     critic_target = Critic(
   208:         n_obs=n_obs, n_act=n_act, num_atoms=args.num_atoms,
   209:         v_min=args.v_min, v_max=args.v_max,
   210:         hidden_dim=args.critic_hidden_dim, device=device,
   211:     )
   212:     critic_target.load_state_dict(critic.state_dict())
   213: 
   214:     actor_optimizer = optim.AdamW(
   215:         actor.parameters(),
   216:         lr=torch.tensor(args.actor_learning_rate, device=device),
   217:         weight_decay=args.weight_decay,
   218:     )
   219:     critic_optimizer = optim.AdamW(
   220:         critic.parameters(),
   221:         lr=torch.tensor(args.critic_learning_rate, device=device),
   222:         weight_decay=args.weight_decay,
   223:     )
   224:     actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
   225:         actor_optimizer, T_max=args.total_timesteps,
   226:         eta_min=torch.tensor(args.actor_learning_rate_end, device=device),
   227:     )
   228:     critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(
   229:         critic_optimizer, T_max=args.total_timesteps,
   230:         eta_min=torch.tensor(args.critic_learning_rate_end, device=device),
   231:     )
   232: 
   233:     return {
   234:         "actor": actor,
   235:         "critic": critic,
   236:         "critic_target": critic_target,
   237:         "actor_optimizer": actor_optimizer,
   238:         "critic_optimizer": critic_optimizer,
   239:         "actor_scheduler": actor_scheduler,
   240:         "critic_scheduler": critic_scheduler,
   241:     }
   242: 
   243: 
   244: def update_critic(data, components, args, scaler, amp_enabled, amp_device_type, amp_dtype):
   245:     """Update the critic network(s).
   246: 
   247:     Modify the critic loss, target computation, or add auxiliary objectives.
   248:     Consider: different distributional RL losses, n-step returns, reward shaping, etc.
   249:     """
   250:     actor = components["actor"]
   251:     critic = components["critic"]
   252:     critic_target = components["critic_target"]
   253:     critic_optimizer = components["critic_optimizer"]
   254: 
   255:     with autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
   256:         observations = data["observations"]
   257:         next_observations = data["next"]["observations"]
   258:         actions = data["actions"]
   259:         rewards = data["next"]["rewards"]
   260:         dones = data["next"]["dones"].bool()
   261:         truncations = data["next"]["truncations"].bool()
   262:         bootstrap = (truncations | ~dones).float()
   263: 
   264:         clipped_noise = torch.randn_like(actions)
   265:         clipped_noise = clipped_noise.mul(args.policy_noise).clamp(-args.noise_clip, args.noise_clip)
   266:         next_state_actions = (actor(next_observations) + clipped_noise).clamp(-1.0, 1.0)
   267:         discount = args.gamma ** data["next"]["effective_n_steps"]
   268: 
   269:         with torch.no_grad():
   270:             qf1_next_proj, qf2_next_proj = critic_target.projection(
   271:                 next_observations, next_state_actions, rewards, bootstrap, discount,
   272:             )
   273:             qf1_next_val = critic_target.get_value(qf1_next_proj)
   274:             qf2_next_val = critic_target.get_value(qf2_next_proj)
   275:             qf_next_dist = torch.where(
   276:                 qf1_next_val.unsqueeze(1) < qf2_next_val.unsqueeze(1),
   277:                 qf1_next_proj, qf2_next_proj,
   278:             )
   279:             qf1_next_dist = qf2_next_dist = qf_next_dist
   280: 
   281:         qf1, qf2 = critic(observations, actions)
   282:         qf1_loss = -torch.sum(qf1_next_dist * F.log_softmax(qf1, dim=1), dim=1).mean()
   283:         qf2_loss = -torch.sum(qf2_next_dist * F.log_softmax(qf2, dim=1), dim=1).mean()
   284:         qf_loss = qf1_loss + qf2_loss
   285: 
   286:     critic_optimizer.zero_grad(set_to_none=True)
   287:     scaler.scale(qf_loss).backward()
   288:     scaler.unscale_(critic_optimizer)
   289:     scaler.step(critic_optimizer)
   290:     scaler.update()
   291: 
   292:     return {"qf_loss": qf_loss.detach(), "qf1_next_val": qf1_next_val}
   293: 
   294: 
   295: def update_actor(data, components, args, scaler, amp_enabled, amp_device_type, amp_dtype):
   296:     """Update the actor (policy) network.
   297: 
   298:     Modify the policy objective, add entropy regularization, or implement
   299:     other policy improvement techniques.
   300:     """
   301:     actor = components["actor"]
   302:     critic = components["critic"]
   303:     actor_optimizer = components["actor_optimizer"]
   304: 
   305:     with autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
   306:         qf1, qf2 = critic(data["observations"], actor(data["observations"]))
   307:         qf1_value = critic.get_value(F.softmax(qf1, dim=1))
   308:         qf2_value = critic.get_value(F.softmax(qf2, dim=1))
   309:         qf_value = torch.minimum(qf1_value, qf2_value)
   310:         actor_loss = -qf_value.mean()
   311: 
   312:     actor_optimizer.zero_grad(set_to_none=True)
   313:     scaler.scale(actor_loss).backward()
   314:     scaler.unscale_(actor_optimizer)
   315:     scaler.step(actor_optimizer)
   316:     scaler.update()
   317: 
   318:     return {"actor_loss": actor_loss.detach()}
   319: 
   320: 
   321: @torch.no_grad()
   322: def soft_update(src, tgt, tau):
   323:     """Soft update target network parameters."""
   324:     src_ps = [p.data for p in src.parameters()]
   325:     tgt_ps = [p.data for p in tgt.parameters()]
   326:     torch._foreach_mul_(tgt_ps, 1.0 - tau)
   327:     torch._foreach_add_(tgt_ps, src_ps, alpha=tau)
   328: 
   329: 
   330: # ═══════════════════════════════════════════════════════════════════════
   331: # ██ EDITABLE SECTION END
   332: # ═══════════════════════════════════════════════════════════════════════
   333: 
   334: 
   335: # ─── FIXED: Argument parsing ───────────────────────────────────────────
   336: def get_args():
   337:     parser = argparse.ArgumentParser()
   338:     parser.add_argument("--env_name", type=str, default="h1hand-stand-v0")
   339:     parser.add_argument("--seed", type=int, default=1)
   340:     parser.add_argument("--total_timesteps", type=int, default=100000)
   341:     parser.add_argument("--num_envs", type=int, default=128)
   342:     parser.add_argument("--batch_size", type=int, default=32768)
   343:     parser.add_argument("--buffer_size", type=int, default=1024 * 50)
   344:     parser.add_argument("--gamma", type=float, default=0.99)
   345:     parser.add_argument("--tau", type=float, default=0.1)
   346:     parser.add_argument("--policy_noise", type=float, default=0.001)
   347:     parser.add_argument("--noise_clip", type=float, default=0.5)
   348:     parser.add_argument("--learning_starts", type=int, default=10)
   349:     parser.add_argument("--policy_frequency", type=int, default=2)
   350:     parser.add_argument("--num_updates", type=int, default=2)
   351:     parser.add_argument("--num_steps", type=int, default=1)
   352:     parser.add_argument("--eval_interval", type=int, default=5000)
   353:     # Network
   354:     parser.add_argument("--actor_hidden_dim", type=int, default=512)
   355:     parser.add_argument("--critic_hidden_dim", type=int, default=1024)
   356:     parser.add_argument("--init_scale", type=float, default=0.01)
   357:     parser.add_argument("--num_atoms", type=int, default=101)
   358:     parser.add_argument("--v_min", type=float, default=-250.0)
   359:     parser.add_argument("--v_max", type=float, default=250.0)
   360:     # Exploration
   361:     parser.add_argument("--std_min", type=float, default=0.001)
   362:     parser.add_argument("--std_max", type=float, default=0.4)
   363:     # Optimizer
   364:     parser.add_argument("--actor_learning_rate", type=float, default=3e-4)
   365:     parser.add_argument("--critic_learning_rate", type=float, default=3e-4)
   366:     parser.add_argument("--actor_learning_rate_end", type=float, default=3e-4)
   367:     parser.add_argument("--critic_learning_rate_end", type=float, default=3e-4)
   368:     parser.add_argument("--weight_decay", type=float, default=0.1)
   369:     # AMP
   370:     parser.add_argument("--amp", action="store_true", default=True)
   371:     parser.add_argument("--amp_dtype", type=str, default="bf16")
   372:     # Misc
   373:     parser.add_argument("--obs_normalization", action="store_true", default=True)
   374:     parser.add_argument("--compile", action="store_true", default=True)
   375:     parser.add_argument("--compile_mode", type=str, default="reduce-overhead")
   376:     parser.add_argument("--device_rank", type=int, default=0)
   377:     return parser.parse_args()
   378: 
   379: 
   380: # ─── FIXED: Main training loop ────────────────────────────────────────
   381: def main():
   382:     args = get_args()
   383:     print(f"Args: {args}")
   384: 
   385:     amp_enabled = args.amp and torch.cuda.is_available()
   386:     amp_device_type = "cuda" if torch.cuda.is_available() else "cpu"
   387:     amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
   388:     scaler = GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)
   389: 
   390:     random.seed(args.seed)
   391:     np.random.seed(args.seed)
   392:     torch.manual_seed(args.seed)
   393:     torch.backends.cudnn.deterministic = True
   394: 
   395:     device = torch.device(f"cuda:{args.device_rank}" if torch.cuda.is_available() else "cpu")
   396:     print(f"Using device: {device}")
   397: 
   398:     # ─── Environment setup (FIXED) ────────────────────────────────────
   399:     from environments.humanoid_bench_env import HumanoidBenchEnv
   400:     envs = HumanoidBenchEnv(args.env_name, args.num_envs, device=device)
   401:     eval_envs = envs
   402: 
   403:     n_act = envs.num_actions
   404:     n_obs = envs.num_obs if type(envs.num_obs) == int else envs.num_obs[0]
   405: 
   406:     if args.obs_normalization:
   407:         obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)
   408:     else:
   409:         obs_normalizer = nn.Identity()
   410: 
   411:     # ─── Build algorithm (EDITABLE function) ──────────────────────────
   412:     components = build_algorithm(n_obs, n_act, args.num_envs, device, args)
   413:     actor = components["actor"]
   414:     critic = components["critic"]
   415:     critic_target = components["critic_target"]
   416: 
   417:     # ─── Replay buffer (FIXED) ────────────────────────────────────────
   418:     rb = SimpleReplayBuffer(
   419:         n_env=args.num_envs, buffer_size=args.buffer_size,
   420:         n_obs=n_obs, n_act=n_act, n_critic_obs=n_obs,
   421:         asymmetric_obs=False, n_steps=args.num_steps,
   422:         gamma=args.gamma, device=device,
   423:     )
   424: 
   425:     # ─── Compile (FIXED) ──────────────────────────────────────────────
   426:     policy = actor.explore
   427:     normalize_obs = obs_normalizer.forward
   428:     if args.compile:
   429:         policy = torch.compile(policy, mode=None)
   430:         normalize_obs = torch.compile(obs_normalizer.forward, mode=None)
   431: 
   432:     # ─── Evaluation (FIXED) ───────────────────────────────────────────
   433:     def evaluate():
   434:         num_eval_envs = eval_envs.num_envs
   435:         episode_returns = torch.zeros(num_eval_envs, device=device)
   436:         done_masks = torch.zeros(num_eval_envs, dtype=torch.bool, device=device)
   437:         obs = eval_envs.reset()
   438:         for i in range(eval_envs.max_episode_steps):
   439:             with torch.no_grad(), autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
   440:                 obs_norm = normalize_obs(obs, update=False)
   441:                 actions = actor(obs_norm)
   442:             next_obs, rewards, dones, infos = eval_envs.step(actions.float())
   443:             episode_returns = torch.where(~done_masks, episode_returns + rewards, episode_returns)
   444:             done_masks = torch.logical_or(done_masks, dones)
   445:             if done_masks.all():
   446:                 break
   447:             obs = next_obs
   448:         return episode_returns.mean().item()
   449: 
   450:     # ─── Training loop (FIXED structure, calls EDITABLE update functions) ─
   451:     obs = envs.reset()
   452:     dones = None
   453:     global_step = 0
   454:     pbar = tqdm.tqdm(total=args.total_timesteps)
   455:     start_time = None
   456:     measure_burnin = 3
   457:     eval_results = []
   458: 
   459:     while global_step < args.total_timesteps:
   460:         mark_step()
   461: 
   462:         if start_time is None and global_step >= measure_burnin + args.learning_starts:
   463:             start_time = time.time()
   464:             measure_start = global_step
   465: 
   466:         # Collect experience
   467:         with torch.no_grad(), autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
   468:             norm_obs = normalize_obs(obs)
   469:             actions = policy(obs=norm_obs, dones=dones)
   470: 
   471:         next_obs, rewards, dones, infos = envs.step(actions.float())
   472:         truncations = infos["time_outs"]
   473: 
   474:         true_next_obs = torch.where(
   475:             dones[:, None] > 0, infos["observations"]["raw"]["obs"], next_obs
   476:         )
   477:         transition = TensorDict({
   478:             "observations": obs,
   479:             "actions": torch.as_tensor(actions, device=device, dtype=torch.float),
   480:             "next": {
   481:                 "observations": true_next_obs,
   482:                 "rewards": torch.as_tensor(rewards, device=device, dtype=torch.float),
   483:                 "truncations": truncations.long(),
   484:                 "dones": dones.long(),
   485:             },
   486:         }, batch_size=(envs.num_envs,), device=device)
   487:         rb.extend(transition)
   488:         obs = next_obs
   489: 
   490:         # Update
   491:         if global_step > args.learning_starts:
   492:             for i in range(args.num_updates):
   493:                 data = rb.sample(max(1, args.batch_size // args.num_envs))
   494:                 data["observations"] = normalize_obs(data["observations"])
   495:                 data["next"]["observations"] = normalize_obs(data["next"]["observations"])
   496: 
   497:                 critic_info = update_critic(data, components, args, scaler, amp_enabled, amp_device_type, amp_dtype)
   498: 
   499:                 should_update_actor = (
   500:                     (args.num_updates > 1 and i % args.policy_frequency == 1)

[truncated: showing at most 500 lines / 60000 bytes from FastTD3/fast_td3/custom_algorithm.py]
```




## How You Will Be Evaluated

After you finish, evaluation runs a fixed set of scripts and aggregates the
metrics they emit. These scripts are **not** in your workspace — you cannot
read or modify them. The labels below indicate what each evaluation tests:

- **h1hand-stand-v0** — wall-clock budget `3:00:00`, compute share `0.5`
- **h1hand-walk-v0** — wall-clock budget `3:00:00`, compute share `0.5`
- **h1hand-run-v0** — wall-clock budget `3:00:00`, compute share `0.5`


Scoring uses the same `combined_score` aggregation as the MLS-Bench
leaderboard. Multiple seeds are averaged.





## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
