# MLS-Bench: robo-diffusion-policy

# Robo-Diffusion: Policy Algorithm Design

## Objective
Design a single model-free offline RL policy algorithm that uses a diffusion actor for action generation and improves offline RL control performance on continuous-control offline RL benchmarks.

This task is intentionally separate from trajectory-diffusion planning. The agent should modify the policy-level actor / critic learning rule, Q / value estimation, or inference-time action selection for a Markov policy. It should not turn the solution into a trajectory planner, classifier-guided planner, or environment-specific evaluation shortcut.

## Background
Diffusion actors parameterize the action distribution as a denoising model conditioned on the state, replacing the unimodal Gaussian heads typical in actor-critic methods. The setup builds on **CleanDiffuser** (Dong et al., NeurIPS 2024, arXiv:2406.09509), a modular diffusion library for decision making, and on the **D4RL** offline RL benchmark (Fu et al., 2020, arXiv:2004.07219). Key paradigms include:
- **Diffusion Q-Learning (DQL)**: BC + Q-maximization on a diffusion actor with twin Q critics; reranks `K` candidate actions at inference.
- **Implicit Diffusion Q-Learning (IDQL)**: decouples actor and critic, trains an IQL-style expectile critic, and reweights candidate actions by a softmax over advantages at inference.
- **Diffusion Policy**: pure behavior cloning with a diffusion actor and single-action sampling at inference.

## What You Can Modify
- Policy algorithm core logic
- Q-function design (if used)
- Action generation strategy
- Training objective
- Actor-critic architecture

## What Is Fixed
- D4RL dataset construction, environment names, and evaluation loop
- Random seeds, episode count, vectorized environment count, and checkpoint names
- The overall offline RL setup: train from fixed D4RL buffers, then evaluate a Markov policy that maps current observation to one action

## Baselines

### default — Diffusion Q-Learning (DQL)
The unmodified template ports CleanDiffuser's `dql_d4rl_mujoco.py` line-for-line: diffusion actor + twin Q critic, BC + Q loss. Reference: Wang, Hunt, Zhou, "Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning", ICLR 2023 (arXiv:2208.06193).

### idql — Implicit Diffusion Q-Learning
Decoupled actor / critic with τ-expectile IQL critic and softmax(adv * β) action reweighting at inference. Reference: Hansen-Estruch et al., 2023 (arXiv:2304.10573); built on IQL (Kostrikov, Nair, Levine, ICLR 2022, arXiv:2110.06169).

### diffusion_policy — Diffusion Policy
Pure behavior cloning with a diffusion actor (no critic, single-action sampling at inference). Reference: Chi et al., RSS 2023 / IJRR 2024 (arXiv:2303.04137).

## Fixed Pipeline

The evaluation loop, environment names, dataset construction, and inference harness are fixed and may not be modified. Training runs for a fixed number of gradient steps; the model may shorten training inside its own edits but cannot exceed the configured limit. The `gradient_steps`, `num_candidates`, `num_envs`, `num_episodes`, and `use_ema` settings are all controlled by the harness config and must not be changed outside the editable region.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/CleanDiffuser/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `CleanDiffuser/pipelines/custom_policy.py`
- editable lines **1–20**
- editable lines **38–165**
- editable lines **169–176**
- editable lines **182–182**
- editable lines **188–207**




## Readable Context


### `CleanDiffuser/pipelines/custom_policy.py`  [EDITABLE — lines 1–20, lines 38–165, lines 169–176, lines 182–182, lines 188–207 only]

```python
     1: import os
     2: from copy import deepcopy
     3: 
     4: import d4rl
     5: import gym
     6: import hydra
     7: import numpy as np
     8: import torch
     9: import torch.nn.functional as F
    10: from torch.optim.lr_scheduler import CosineAnnealingLR
    11: from torch.utils.data import DataLoader
    12: 
    13: from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoTDDataset
    14: from cleandiffuser.dataset.dataset_utils import loop_dataloader
    15: from cleandiffuser.diffusion import DiscreteDiffusionSDE
    16: from cleandiffuser.nn_condition import IdentityCondition
    17: from cleandiffuser.nn_diffusion import DQLMlp
    18: from cleandiffuser.utils import report_parameters, DQLCritic, FreezeModules
    19: from utils import set_seed
    20: 
    21: 
    22: @hydra.main(config_path="../configs/custom/mujoco", config_name="mujoco", version_base=None)
    23: def pipeline(args):
    24: 
    25:     set_seed(args.seed)
    26: 
    27:     save_path = f'results/{args.pipeline_name}/{args.task.env_name}/'
    28:     if os.path.exists(save_path) is False:
    29:         os.makedirs(save_path)
    30: 
    31:     # ---------------------- Create Dataset ----------------------
    32:     env = gym.make(args.task.env_name)
    33:     dataset = D4RLMuJoCoTDDataset(d4rl.qlearning_dataset(env), args.normalize_reward)
    34:     dataloader = DataLoader(
    35:         dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    36:     obs_dim, act_dim = dataset.o_dim, dataset.a_dim
    37: 
    38:     # ============================================================================
    39:     # EDITABLE REGION: Policy Algorithm (lines 40-205)
    40:     # ============================================================================
    41:     # Defines the actor (diffusion policy), optional critic(s), training loop,
    42:     # and inference action-selection. The template defaults to Diffusion
    43:     # Q-Learning (DQL): diffusion actor + twin Q critic with BC + Q loss.
    44:     # Baselines may swap in IQL Q+V (idql) or strip the critic entirely
    45:     # (diffusion_policy = pure BC).
    46: 
    47:     # --------------- Network Architecture -----------------
    48:     nn_diffusion = DQLMlp(obs_dim, act_dim, emb_dim=64, timestep_emb_type="positional").to(args.device)
    49:     nn_condition = IdentityCondition(dropout=0.0).to(args.device)
    50: 
    51:     print(f"======================= Parameter Report of Diffusion Model =======================")
    52:     report_parameters(nn_diffusion)
    53:     print(f"==============================================================================")
    54: 
    55:     # --------------- Diffusion Model Actor --------------------
    56:     actor = DiscreteDiffusionSDE(
    57:         nn_diffusion, nn_condition, predict_noise=args.predict_noise, optim_params={"lr": args.actor_learning_rate},
    58:         x_max=+1. * torch.ones((1, act_dim), device=args.device),
    59:         x_min=-1. * torch.ones((1, act_dim), device=args.device),
    60:         diffusion_steps=args.diffusion_steps, ema_rate=args.ema_rate, device=args.device)
    61: 
    62:     # ------------------ Critic ---------------------
    63:     critic = DQLCritic(obs_dim, act_dim, hidden_dim=args.hidden_dim).to(args.device)
    64:     critic_target = deepcopy(critic).requires_grad_(False).eval()
    65:     critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_learning_rate)
    66: 
    67:     # ---------------------- Training ----------------------
    68:     if args.mode == "train":
    69: 
    70:         actor_lr_scheduler = CosineAnnealingLR(actor.optimizer, T_max=args.gradient_steps)
    71:         critic_lr_scheduler = CosineAnnealingLR(critic_optim, T_max=args.gradient_steps)
    72: 
    73:         actor.train()
    74:         critic.train()
    75: 
    76:         n_gradient_step = 0
    77:         log = {"bc_loss": 0., "q_loss": 0., "critic_loss": 0., "target_q_mean": 0.}
    78: 
    79:         prior = torch.zeros((args.batch_size, act_dim), device=args.device)
    80: 
    81:         for batch in loop_dataloader(dataloader):
    82: 
    83:             obs, next_obs = batch["obs"]["state"].to(args.device), batch["next_obs"]["state"].to(args.device)
    84:             act = batch["act"].to(args.device)
    85:             rew = batch["rew"].to(args.device)
    86:             tml = batch["tml"].to(args.device)
    87: 
    88:             # Critic Training
    89:             current_q1, current_q2 = critic(obs, act)
    90: 
    91:             next_act, _ = actor.sample(
    92:                 prior, solver=args.solver,
    93:                 n_samples=args.batch_size, sample_steps=args.sampling_steps, use_ema=True,
    94:                 temperature=1.0, condition_cfg=next_obs, w_cfg=1.0, requires_grad=False)
    95: 
    96:             target_q = torch.min(*critic_target(next_obs, next_act))
    97:             target_q = (rew + (1 - tml) * args.discount * target_q).detach()
    98: 
    99:             critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
   100: 
   101:             critic_optim.zero_grad()
   102:             critic_loss.backward()
   103:             critic_optim.step()
   104: 
   105:             # Policy Training
   106:             bc_loss = actor.loss(act, obs)
   107:             new_act, _ = actor.sample(
   108:                 prior, solver=args.solver,
   109:                 n_samples=args.batch_size, sample_steps=args.sampling_steps, use_ema=False,
   110:                 temperature=1.0, condition_cfg=obs, w_cfg=1.0, requires_grad=True)
   111: 
   112:             with FreezeModules([critic, ]):
   113:                 q1_new_action, q2_new_action = critic(obs, new_act)
   114:             if np.random.uniform() > 0.5:
   115:                 q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
   116:             else:
   117:                 q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
   118:             actor_loss = bc_loss + args.task.eta * q_loss
   119: 
   120:             actor.optimizer.zero_grad()
   121:             actor_loss.backward()
   122:             actor.optimizer.step()
   123: 
   124:             actor_lr_scheduler.step()
   125:             critic_lr_scheduler.step()
   126: 
   127:             # ema
   128:             if n_gradient_step % args.ema_update_interval == 0:
   129:                 if n_gradient_step >= 1000:
   130:                     actor.ema_update()
   131:                 for param, target_param in zip(critic.parameters(), critic_target.parameters()):
   132:                     target_param.data.copy_(0.995 * param.data + (1 - 0.995) * target_param.data)
   133: 
   134:             log["bc_loss"] += bc_loss.item()
   135:             log["q_loss"] += q_loss.item()
   136:             log["critic_loss"] += critic_loss.item()
   137:             log["target_q_mean"] += target_q.mean().item()
   138: 
   139:             if (n_gradient_step + 1) % args.log_interval == 0:
   140:                 log["gradient_steps"] = n_gradient_step + 1
   141:                 log["bc_loss"] /= args.log_interval
   142:                 log["q_loss"] /= args.log_interval
   143:                 log["critic_loss"] /= args.log_interval
   144:                 log["target_q_mean"] /= args.log_interval
   145:                 print(f"TRAIN_METRICS gradient_steps={log['gradient_steps']} "
   146:                       f"bc_loss={log['bc_loss']:.4f} q_loss={log['q_loss']:.4f} "
   147:                       f"critic_loss={log['critic_loss']:.4f} target_q_mean={log['target_q_mean']:.4f}")
   148:                 log = {"bc_loss": 0., "q_loss": 0., "critic_loss": 0., "target_q_mean": 0.}
   149: 
   150:             if (n_gradient_step + 1) % args.save_interval == 0:
   151:                 actor.save(save_path + f"diffusion_ckpt_{n_gradient_step + 1}.pt")
   152:                 actor.save(save_path + f"diffusion_ckpt_latest.pt")
   153:                 torch.save({
   154:                     "critic": critic.state_dict(),
   155:                     "critic_target": critic_target.state_dict(),
   156:                 }, save_path + f"critic_ckpt_{n_gradient_step + 1}.pt")
   157:                 torch.save({
   158:                     "critic": critic.state_dict(),
   159:                     "critic_target": critic_target.state_dict(),
   160:                 }, save_path + f"critic_ckpt_latest.pt")
   161: 
   162:             n_gradient_step += 1
   163:             if n_gradient_step >= args.gradient_steps:
   164:                 break
   165: 
   166:     # ---------------------- Inference ----------------------
   167:     elif args.mode == "inference":
   168: 
   169:         actor.load(save_path + f"diffusion_ckpt_{args.ckpt}.pt")
   170:         critic_ckpt = torch.load(save_path + f"critic_ckpt_{args.ckpt}.pt")
   171:         critic.load_state_dict(critic_ckpt["critic"])
   172:         critic_target.load_state_dict(critic_ckpt["critic_target"])
   173: 
   174:         actor.eval()
   175:         critic.eval()
   176:         critic_target.eval()
   177: 
   178:         env_eval = gym.vector.make(args.task.env_name, args.num_envs)
   179:         normalizer = dataset.get_normalizer()
   180:         episode_rewards = []
   181: 
   182:         prior = torch.zeros((args.num_envs * args.num_candidates, act_dim), device=args.device)
   183:         for i in range(args.num_episodes):
   184: 
   185:             env_eval.seed(args.seed + i * args.num_envs) if hasattr(env_eval, "seed") else None; obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0
   186: 
   187:             while not np.all(cum_done) and t < 1000 + 1:
   188:                 obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
   189:                 obs = obs.unsqueeze(1).repeat(1, args.num_candidates, 1).view(-1, obs_dim)
   190: 
   191:                 act, log = actor.sample(
   192:                     prior,
   193:                     solver=args.solver,
   194:                     n_samples=args.num_envs * args.num_candidates,
   195:                     sample_steps=args.sampling_steps,
   196:                     condition_cfg=obs, w_cfg=1.0,
   197:                     use_ema=args.use_ema, temperature=args.temperature)
   198: 
   199:                 with torch.no_grad():
   200:                     q = critic_target.q_min(obs, act)
   201:                     q = q.view(-1, args.num_candidates, 1)
   202:                     w = torch.softmax(q * args.task.weight_temperature, 1)
   203:                     act = act.view(-1, args.num_candidates, act_dim)
   204: 
   205:                     indices = torch.multinomial(w.squeeze(-1), 1).squeeze(-1)
   206:                     sampled_act = act[torch.arange(act.shape[0]), indices].cpu().numpy()
   207: 
   208:                 obs, rew, done, info = env_eval.step(sampled_act)
   209: 
   210:                 t += 1
   211:                 cum_done = done if cum_done is None else np.logical_or(cum_done, done)
   212:                 ep_reward += (rew * (1 - cum_done)) if t < 1000 else rew
   213: 
   214:                 if np.all(cum_done):
   215:                     break
   216: 
   217:             episode_rewards.append(ep_reward)
   218: 
   219:         raw_episode_rewards = episode_rewards
   220:         episode_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
   221:         episode_rewards = np.array(episode_rewards)
   222:         mean_score = float(np.mean(episode_rewards))
   223:         std_score = float(np.std(episode_rewards))
   224:         mean_ep_reward = float(np.mean(raw_episode_rewards))
   225:         print(f"EVAL_METRICS normalized_score={mean_score:.4f} normalized_score_std={std_score:.4f} episode_reward={mean_ep_reward:.2f}")
   226: 
   227:     else:
   228:         raise ValueError(f"Invalid mode: {args.mode}")
   229: 
   230: 
   231: if __name__ == "__main__":
   232:     pipeline()
```


## Adapter Warnings

Some reference context could not be rendered completely:

- `default` has no edit_ops entry

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `idql` baseline — editable region  [READ-ONLY — reference implementation]

In `CleanDiffuser/pipelines/custom_policy.py`:

```python
Lines 1–18:
     1: import os
     2: from copy import deepcopy
     3: 
     4: import d4rl
     5: import gym
     6: import hydra
     7: import numpy as np
     8: import torch
     9: from torch.optim.lr_scheduler import CosineAnnealingLR
    10: from torch.utils.data import DataLoader
    11: 
    12: from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoTDDataset
    13: from cleandiffuser.dataset.dataset_utils import loop_dataloader
    14: from cleandiffuser.diffusion import DiscreteDiffusionSDE
    15: from cleandiffuser.nn_condition import IdentityCondition
    16: from cleandiffuser.nn_diffusion import IDQLMlp
    17: from cleandiffuser.utils import report_parameters, IDQLQNet, IDQLVNet
    18: from utils import set_seed
    19: 
    20: @hydra.main(config_path="../configs/custom/mujoco", config_name="mujoco", version_base=None)
    21: def pipeline(args):

Lines 36–145:
    33:         dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    34:     obs_dim, act_dim = dataset.o_dim, dataset.a_dim
    35: 
    36:     # ============================================================================
    37:     # idql baseline: full IDQL pipeline (Hansen-Estruch et al., 2023)
    38:     # ============================================================================
    39: 
    40:     # --------------- Network Architecture -----------------
    41:     nn_diffusion = IDQLMlp(
    42:         obs_dim, act_dim, emb_dim=64,
    43:         hidden_dim=args.actor_hidden_dim, n_blocks=args.actor_n_blocks, dropout=args.actor_dropout,
    44:         timestep_emb_type="positional")
    45:     nn_condition = IdentityCondition(dropout=0.0)
    46: 
    47:     print(f"======================= Parameter Report of Diffusion Model =======================")
    48:     report_parameters(nn_diffusion)
    49:     print(f"==============================================================================")
    50: 
    51:     # --------------- Diffusion Model Actor --------------------
    52:     actor = DiscreteDiffusionSDE(
    53:         nn_diffusion, nn_condition, predict_noise=args.predict_noise, optim_params={"lr": args.actor_learning_rate},
    54:         x_max=+1. * torch.ones((1, act_dim)),
    55:         x_min=-1. * torch.ones((1, act_dim)),
    56:         diffusion_steps=args.diffusion_steps, ema_rate=args.ema_rate, device=args.device)
    57: 
    58:     # ------------------ Critic ---------------------
    59:     iql_q = IDQLQNet(obs_dim, act_dim, hidden_dim=args.critic_hidden_dim).to(args.device)
    60:     iql_q_target = deepcopy(iql_q).requires_grad_(False).eval()
    61:     iql_v = IDQLVNet(obs_dim, hidden_dim=args.critic_hidden_dim).to(args.device)
    62: 
    63:     q_optim = torch.optim.Adam(iql_q.parameters(), lr=args.critic_learning_rate)
    64:     v_optim = torch.optim.Adam(iql_v.parameters(), lr=args.critic_learning_rate)
    65: 
    66:     # ---------------------- Training ----------------------
    67:     if args.mode == "train":
    68: 
    69:         actor_lr_scheduler = CosineAnnealingLR(actor.optimizer, T_max=args.gradient_steps)
    70:         q_lr_scheduler = CosineAnnealingLR(q_optim, T_max=args.gradient_steps)
    71:         v_lr_scheduler = CosineAnnealingLR(v_optim, T_max=args.gradient_steps)
    72: 
    73:         actor.train()
    74:         iql_q.train()
    75:         iql_v.train()
    76: 
    77:         n_gradient_step = 0
    78:         log = {"bc_loss": 0., "q_loss": 0., "v_loss": 0.}
    79: 
    80:         for batch in loop_dataloader(dataloader):
    81: 
    82:             obs, next_obs = batch["obs"]["state"].to(args.device), batch["next_obs"]["state"].to(args.device)
    83:             act = batch["act"].to(args.device)
    84:             rew = batch["rew"].to(args.device)
    85:             tml = batch["tml"].to(args.device)
    86: 
    87:             # -- IQL Training
    88:             if n_gradient_step % 2 == 0:
    89: 
    90:                 q = iql_q_target(obs, act)
    91:                 v = iql_v(obs)
    92:                 v_loss = (torch.abs(args.iql_tau - ((q - v) < 0).float()) * (q - v) ** 2).mean()
    93: 
    94:                 v_optim.zero_grad()
    95:                 v_loss.backward()
    96:                 v_optim.step()
    97: 
    98:                 with torch.no_grad():
    99:                     td_target = rew + args.discount * (1 - tml) * iql_v(next_obs)
   100:                 q1, q2 = iql_q.both(obs, act)
   101:                 q_loss = ((q1 - td_target) ** 2 + (q2 - td_target) ** 2).mean()
   102:                 q_optim.zero_grad()
   103:                 q_loss.backward()
   104:                 q_optim.step()
   105: 
   106:                 q_lr_scheduler.step()
   107:                 v_lr_scheduler.step()
   108: 
   109:                 for param, target_param in zip(iql_q.parameters(), iql_q_target.parameters()):
   110:                     target_param.data.copy_(0.995 * param.data + (1 - 0.995) * target_param.data)
   111: 
   112:             # -- Policy Training
   113:             bc_loss = actor.update(act, obs)["loss"]
   114:             actor_lr_scheduler.step()
   115: 
   116:             log["bc_loss"] += bc_loss
   117:             log["q_loss"] += q_loss.item()
   118:             log["v_loss"] += v_loss.item()
   119: 
   120:             if (n_gradient_step + 1) % args.log_interval == 0:
   121:                 log["gradient_steps"] = n_gradient_step + 1
   122:                 log["bc_loss"] /= args.log_interval
   123:                 log["q_loss"] /= args.log_interval
   124:                 log["v_loss"] /= args.log_interval
   125:                 print(f"TRAIN_METRICS gradient_steps={log['gradient_steps']} "
   126:                       f"bc_loss={log['bc_loss']:.4f} q_loss={log['q_loss']:.4f} v_loss={log['v_loss']:.4f}")
   127:                 log = {"bc_loss": 0., "q_loss": 0., "v_loss": 0.}
   128: 
   129:             if (n_gradient_step + 1) % args.save_interval == 0:
   130:                 actor.save(save_path + f"diffusion_ckpt_{n_gradient_step + 1}.pt")
   131:                 actor.save(save_path + f"diffusion_ckpt_latest.pt")
   132:                 torch.save({
   133:                     "iql_q": iql_q.state_dict(),
   134:                     "iql_q_target": iql_q_target.state_dict(),
   135:                     "iql_v": iql_v.state_dict(),
   136:                 }, save_path + f"iql_ckpt_{n_gradient_step + 1}.pt")
   137:                 torch.save({
   138:                     "iql_q": iql_q.state_dict(),
   139:                     "iql_q_target": iql_q_target.state_dict(),
   140:                     "iql_v": iql_v.state_dict(),
   141:                 }, save_path + f"iql_ckpt_latest.pt")
   142: 
   143:             n_gradient_step += 1
   144:             if n_gradient_step >= args.gradient_steps:
   145:                 break
   146:     # ---------------------- Inference ----------------------
   147:     elif args.mode == "inference":
   148: 

Lines 149–158:
   146:     # ---------------------- Inference ----------------------
   147:     elif args.mode == "inference":
   148: 
   149:         actor.load(save_path + f"diffusion_ckpt_{args.ckpt}.pt")
   150:         critic_ckpt = torch.load(save_path + f"iql_ckpt_{args.ckpt}.pt")
   151:         iql_q.load_state_dict(critic_ckpt["iql_q"])
   152:         iql_q_target.load_state_dict(critic_ckpt["iql_q_target"])
   153:         iql_v.load_state_dict(critic_ckpt["iql_v"])
   154: 
   155:         actor.eval()
   156:         iql_q.eval()
   157:         iql_q_target.eval()
   158:         iql_v.eval()
   159: 
   160:         env_eval = gym.vector.make(args.task.env_name, args.num_envs)
   161:         normalizer = dataset.get_normalizer()

Lines 164–164:
   161:         normalizer = dataset.get_normalizer()
   162:         episode_rewards = []
   163: 
   164:         prior = torch.zeros((args.num_envs * args.num_candidates, act_dim), device=args.device)
   165:         for i in range(args.num_episodes):
   166: 
   167:             env_eval.seed(args.seed + i * args.num_envs) if hasattr(env_eval, "seed") else None; obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0

Lines 170–189:
   167:             env_eval.seed(args.seed + i * args.num_envs) if hasattr(env_eval, "seed") else None; obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0
   168: 
   169:             while not np.all(cum_done) and t < 1000 + 1:
   170:                 obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
   171:                 obs = obs.unsqueeze(1).repeat(1, args.num_candidates, 1).view(-1, obs_dim)
   172: 
   173:                 act, _ = actor.sample(
   174:                     prior,
   175:                     solver=args.solver,
   176:                     n_samples=args.num_envs * args.num_candidates,
   177:                     sample_steps=args.sampling_steps,
   178:                     condition_cfg=obs, w_cfg=1.0,
   179:                     use_ema=args.use_ema, temperature=args.temperature)
   180: 
   181:                 with torch.no_grad():
   182:                     q = iql_q_target(obs, act)
   183:                     v = iql_v(obs)
   184:                     adv = (q - v).view(-1, args.num_candidates, 1)
   185:                     w = torch.softmax(adv * args.task.weight_temperature, 1)
   186:                     act = act.view(-1, args.num_candidates, act_dim)
   187:                     p = w / w.sum(1, keepdim=True)
   188:                     indices = torch.multinomial(p.squeeze(-1), 1).squeeze(-1)
   189:                     sampled_act = act[torch.arange(act.shape[0]), indices].cpu().numpy()
   190:                 obs, rew, done, info = env_eval.step(sampled_act)
   191: 
   192:                 t += 1
```

### `diffusion_policy` baseline — editable region  [READ-ONLY — reference implementation]

In `CleanDiffuser/pipelines/custom_policy.py`:

```python
Lines 1–20:
     1: import os
     2: from copy import deepcopy
     3: 
     4: import d4rl
     5: import gym
     6: import hydra
     7: import numpy as np
     8: import torch
     9: import torch.nn.functional as F
    10: from torch.optim.lr_scheduler import CosineAnnealingLR
    11: from torch.utils.data import DataLoader
    12: 
    13: from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoTDDataset
    14: from cleandiffuser.dataset.dataset_utils import loop_dataloader
    15: from cleandiffuser.diffusion import DiscreteDiffusionSDE
    16: from cleandiffuser.nn_condition import IdentityCondition
    17: from cleandiffuser.nn_diffusion import DQLMlp
    18: from cleandiffuser.utils import report_parameters, DQLCritic, FreezeModules
    19: from utils import set_seed
    20: 
    21: 
    22: @hydra.main(config_path="../configs/custom/mujoco", config_name="mujoco", version_base=None)
    23: def pipeline(args):

Lines 38–92:
    35:         dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    36:     obs_dim, act_dim = dataset.o_dim, dataset.a_dim
    37: 
    38:     # ============================================================================
    39:     # diffusion_policy baseline: diffusion BC only, no critic / no reranking
    40:     # ============================================================================
    41: 
    42:     # --------------- Network Architecture -----------------
    43:     nn_diffusion = DQLMlp(obs_dim, act_dim, emb_dim=64, timestep_emb_type="positional").to(args.device)
    44:     nn_condition = IdentityCondition(dropout=0.0).to(args.device)
    45: 
    46:     print(f"======================= Parameter Report of Diffusion Model =======================")
    47:     report_parameters(nn_diffusion)
    48:     print(f"==============================================================================")
    49: 
    50:     # --------------- Diffusion Model Actor --------------------
    51:     actor = DiscreteDiffusionSDE(
    52:         nn_diffusion, nn_condition, predict_noise=args.predict_noise, optim_params={"lr": args.actor_learning_rate},
    53:         x_max=+1. * torch.ones((1, act_dim), device=args.device),
    54:         x_min=-1. * torch.ones((1, act_dim), device=args.device),
    55:         diffusion_steps=args.diffusion_steps, ema_rate=args.ema_rate, device=args.device)
    56: 
    57:     # ---------------------- Training ----------------------
    58:     if args.mode == "train":
    59: 
    60:         actor_lr_scheduler = CosineAnnealingLR(actor.optimizer, T_max=args.gradient_steps)
    61: 
    62:         actor.train()
    63: 
    64:         n_gradient_step = 0
    65:         log = {"bc_loss": 0.}
    66: 
    67:         for batch in loop_dataloader(dataloader):
    68: 
    69:             obs = batch["obs"]["state"].to(args.device)
    70:             act = batch["act"].to(args.device)
    71: 
    72:             bc_loss = actor.update(act, obs)["loss"]
    73:             actor_lr_scheduler.step()
    74: 
    75:             if n_gradient_step % args.ema_update_interval == 0 and n_gradient_step >= 1000:
    76:                 actor.ema_update()
    77: 
    78:             log["bc_loss"] += bc_loss
    79: 
    80:             if (n_gradient_step + 1) % args.log_interval == 0:
    81:                 log["gradient_steps"] = n_gradient_step + 1
    82:                 log["bc_loss"] /= args.log_interval
    83:                 print(f"TRAIN_METRICS gradient_steps={log['gradient_steps']} bc_loss={log['bc_loss']:.4f}")
    84:                 log = {"bc_loss": 0.}
    85: 
    86:             if (n_gradient_step + 1) % args.save_interval == 0:
    87:                 actor.save(save_path + f"diffusion_ckpt_{n_gradient_step + 1}.pt")
    88:                 actor.save(save_path + f"diffusion_ckpt_latest.pt")
    89: 
    90:             n_gradient_step += 1
    91:             if n_gradient_step >= args.gradient_steps:
    92:                 break
    93:     # ---------------------- Inference ----------------------
    94:     elif args.mode == "inference":
    95: 

Lines 96–97:
    93:     # ---------------------- Inference ----------------------
    94:     elif args.mode == "inference":
    95: 
    96:         actor.load(save_path + f"diffusion_ckpt_{args.ckpt}.pt")
    97:         actor.eval()
    98: 
    99:         env_eval = gym.vector.make(args.task.env_name, args.num_envs)
   100:         normalizer = dataset.get_normalizer()

Lines 103–103:
   100:         normalizer = dataset.get_normalizer()
   101:         episode_rewards = []
   102: 
   103:         prior = torch.zeros((args.num_envs, act_dim), device=args.device)
   104:         for i in range(args.num_episodes):
   105: 
   106:             env_eval.seed(args.seed + i * args.num_envs) if hasattr(env_eval, "seed") else None; obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0

Lines 109–118:
   106:             env_eval.seed(args.seed + i * args.num_envs) if hasattr(env_eval, "seed") else None; obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0
   107: 
   108:             while not np.all(cum_done) and t < 1000 + 1:
   109:                 obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
   110: 
   111:                 act, _ = actor.sample(
   112:                     prior,
   113:                     solver=args.solver,
   114:                     n_samples=args.num_envs,
   115:                     sample_steps=args.sampling_steps,
   116:                     condition_cfg=obs, w_cfg=1.0,
   117:                     use_ema=args.use_ema, temperature=args.temperature)
   118:                 sampled_act = act.clip(-1., 1.).cpu().numpy()
   119:                 obs, rew, done, info = env_eval.step(sampled_act)
   120: 
   121:                 t += 1
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
