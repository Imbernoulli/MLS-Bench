# MLS-Bench: robo-diffusion-sampling-method

# Robo-Diffusion: Sampling Algorithm Design

## Objective
Design a single efficient diffusion sampler for a fixed DQL-style diffusion policy that achieves high quality at low inference NFE (number of function evaluations).

This task is deliberately about the inference-time reverse process, not policy learning, guidance, or trajectory planning. The trained actor / critic, dataset, environment list, seeds, and evaluation loop are fixed.

## Background
A diffusion policy's wall-clock inference cost is dominated by the number of reverse-process steps. Different ODE / SDE solvers reach a given sample quality at different NFE budgets:
- **DDPM** (Ho, Jain, Abbeel, NeurIPS 2020, arXiv:2006.11239): the original Markovian sampler; high quality but slow.
- **DDIM** (Song, Meng, Ermon, ICLR 2021, arXiv:2010.02502): non-Markovian deterministic sampler that hits comparable quality in 10–50× fewer steps.
- **DPM-Solver++** (Lu et al., 2022, arXiv:2211.01095): high-order ODE solver that reaches strong sample quality at ~10–20 steps for guided DPM sampling.

The setup builds on **CleanDiffuser** (Dong et al., NeurIPS 2024, arXiv:2406.09509) and the underlying actor is a DQL-style diffusion policy (Wang et al., ICLR 2023, arXiv:2208.06193) trained on **D4RL** (Fu et al., 2020, arXiv:2004.07219).

## What You Can Modify
- The **sampling algorithm itself** — the `EDITABLE REGION: Sampling Algorithm` block in `CleanDiffuser/pipelines/custom_sampling_method.py`, which turns a prior and an observation into actions. The default delegates to CleanDiffuser's built-in solvers; you may instead write the reverse process yourself, calling the denoiser (`actor.model_ema["diffusion"]`) directly with whatever discretization, step schedule, order or correction you want.
- `solver` and `sampling_steps` in `CleanDiffuser/configs/custom/mujoco/mujoco.yaml`, which drive the default implementation.

## What Is Fixed
- The actor and critic architectures, the training objective and the training loop
- `diffusion_steps`, training budgets, checkpoint selection, and EMA use
- Candidate selection, the environments, seeds, and vectorized evaluation loop

NFE is **measured, not declared**: a forward hook on the denoiser counts real network evaluations during evaluation and reports the average per action sample. Writing your own reverse process is therefore in scope — a sampler that spends 40 evaluations is scored as 40 no matter what any config field says, and one that reaches the same return in 10 is scored as 10.

## Baselines

### default
DDPM sampling with 100 steps — standard but slow. This is the unmodified
template baseline (registered as `default` in the config).

### ddim
DDIM sampling with 20 steps — faster deterministic sampling.

### dpm_solver
DPM-Solver++ with 10 steps — fast high-quality sampling.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/CleanDiffuser/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits that change code outside these ranges — or creating new files, or
deleting whole files — will cause your submission to be invalid.

The line numbers mark an editable **region**, not a fixed line-count budget: you
may add or remove lines inside it. Only code outside the editable ranges must
stay unchanged.

- `CleanDiffuser/pipelines/custom_sampling_method.py`
- editable lines **213–235**
- `CleanDiffuser/configs/custom/mujoco/mujoco.yaml`
- editable lines **15–15**
- editable lines **17–17**


Other files you may **read** for context (do not modify):
- `CleanDiffuser/pipelines/custom_sampling_method.py`


## Readable Context


### `CleanDiffuser/pipelines/custom_sampling_method.py`  [EDITABLE — lines 213–235 only]

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
    39:     # FIXED: Policy, Critic and Training
    40:     # ============================================================================
    41:     # Diffusion Q-Learning (DQL): diffusion actor + twin Q critic with BC + Q
    42:     # loss. The trained actor/critic, dataset, environment list, seeds and
    43:     # evaluation loop are fixed — this task is about the sampler, so the thing
    44:     # you edit is the reverse process at inference time, further below.
    45: 
    46:     # --------------- Network Architecture -----------------
    47:     nn_diffusion = DQLMlp(obs_dim, act_dim, emb_dim=64, timestep_emb_type="positional").to(args.device)
    48:     nn_condition = IdentityCondition(dropout=0.0).to(args.device)
    49: 
    50:     print(f"======================= Parameter Report of Diffusion Model =======================")
    51:     report_parameters(nn_diffusion)
    52:     print(f"==============================================================================")
    53: 
    54:     # --------------- Diffusion Model Actor --------------------
    55:     actor = DiscreteDiffusionSDE(
    56:         nn_diffusion, nn_condition, predict_noise=args.predict_noise, optim_params={"lr": args.actor_learning_rate},
    57:         x_max=+1. * torch.ones((1, act_dim), device=args.device),
    58:         x_min=-1. * torch.ones((1, act_dim), device=args.device),
    59:         diffusion_steps=args.diffusion_steps, ema_rate=args.ema_rate, device=args.device)
    60: 
    61:     # ------------------ Critic ---------------------
    62:     critic = DQLCritic(obs_dim, act_dim, hidden_dim=args.hidden_dim).to(args.device)
    63:     critic_target = deepcopy(critic).requires_grad_(False).eval()
    64:     critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_learning_rate)
    65: 
    66:     # ---------------------- Training ----------------------
    67:     if args.mode == "train":
    68: 
    69:         actor_lr_scheduler = CosineAnnealingLR(actor.optimizer, T_max=args.gradient_steps)
    70:         critic_lr_scheduler = CosineAnnealingLR(critic_optim, T_max=args.gradient_steps)
    71: 
    72:         actor.train()
    73:         critic.train()
    74: 
    75:         n_gradient_step = 0
    76:         log = {"bc_loss": 0., "q_loss": 0., "critic_loss": 0., "target_q_mean": 0.}
    77: 
    78:         prior = torch.zeros((args.batch_size, act_dim), device=args.device)
    79: 
    80:         for batch in loop_dataloader(dataloader):
    81: 
    82:             obs, next_obs = batch["obs"]["state"].to(args.device), batch["next_obs"]["state"].to(args.device)
    83:             act = batch["act"].to(args.device)
    84:             rew = batch["rew"].to(args.device)
    85:             tml = batch["tml"].to(args.device)
    86: 
    87:             # Critic Training
    88:             current_q1, current_q2 = critic(obs, act)
    89: 
    90:             next_act, _ = actor.sample(
    91:                 prior, solver=args.solver,
    92:                 n_samples=args.batch_size, sample_steps=args.sampling_steps, use_ema=True,
    93:                 temperature=1.0, condition_cfg=next_obs, w_cfg=1.0, requires_grad=False)
    94: 
    95:             target_q = torch.min(*critic_target(next_obs, next_act))
    96:             target_q = (rew + (1 - tml) * args.discount * target_q).detach()
    97: 
    98:             critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
    99: 
   100:             critic_optim.zero_grad()
   101:             critic_loss.backward()
   102:             critic_optim.step()
   103: 
   104:             # Policy Training
   105:             bc_loss = actor.loss(act, obs)
   106:             new_act, _ = actor.sample(
   107:                 prior, solver=args.solver,
   108:                 n_samples=args.batch_size, sample_steps=args.sampling_steps, use_ema=False,
   109:                 temperature=1.0, condition_cfg=obs, w_cfg=1.0, requires_grad=True)
   110: 
   111:             with FreezeModules([critic, ]):
   112:                 q1_new_action, q2_new_action = critic(obs, new_act)
   113:             if np.random.uniform() > 0.5:
   114:                 q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
   115:             else:
   116:                 q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
   117:             actor_loss = bc_loss + args.task.eta * q_loss
   118: 
   119:             actor.optimizer.zero_grad()
   120:             actor_loss.backward()
   121:             actor.optimizer.step()
   122: 
   123:             actor_lr_scheduler.step()
   124:             critic_lr_scheduler.step()
   125: 
   126:             # ema
   127:             if n_gradient_step % args.ema_update_interval == 0:
   128:                 if n_gradient_step >= 1000:
   129:                     actor.ema_update()
   130:                 for param, target_param in zip(critic.parameters(), critic_target.parameters()):
   131:                     target_param.data.copy_(0.995 * param.data + (1 - 0.995) * target_param.data)
   132: 
   133:             log["bc_loss"] += bc_loss.item()
   134:             log["q_loss"] += q_loss.item()
   135:             log["critic_loss"] += critic_loss.item()
   136:             log["target_q_mean"] += target_q.mean().item()
   137: 
   138:             if (n_gradient_step + 1) % args.log_interval == 0:
   139:                 log["gradient_steps"] = n_gradient_step + 1
   140:                 log["bc_loss"] /= args.log_interval
   141:                 log["q_loss"] /= args.log_interval
   142:                 log["critic_loss"] /= args.log_interval
   143:                 log["target_q_mean"] /= args.log_interval
   144:                 print(f"TRAIN_METRICS gradient_steps={log['gradient_steps']} "
   145:                       f"bc_loss={log['bc_loss']:.4f} q_loss={log['q_loss']:.4f} "
   146:                       f"critic_loss={log['critic_loss']:.4f} target_q_mean={log['target_q_mean']:.4f}")
   147:                 log = {"bc_loss": 0., "q_loss": 0., "critic_loss": 0., "target_q_mean": 0.}
   148: 
   149:             if (n_gradient_step + 1) % args.save_interval == 0:
   150:                 actor.save(save_path + f"diffusion_ckpt_{n_gradient_step + 1}.pt")
   151:                 actor.save(save_path + f"diffusion_ckpt_latest.pt")
   152:                 torch.save({
   153:                     "critic": critic.state_dict(),
   154:                     "critic_target": critic_target.state_dict(),
   155:                 }, save_path + f"critic_ckpt_{n_gradient_step + 1}.pt")
   156:                 torch.save({
   157:                     "critic": critic.state_dict(),
   158:                     "critic_target": critic_target.state_dict(),
   159:                 }, save_path + f"critic_ckpt_latest.pt")
   160: 
   161:             n_gradient_step += 1
   162:             if n_gradient_step >= args.gradient_steps:
   163:                 break
   164: 
   165:     # ---------------------- Inference ----------------------
   166:     elif args.mode == "inference":
   167: 
   168:         actor.load(save_path + f"diffusion_ckpt_{args.ckpt}.pt")
   169:         critic_ckpt = torch.load(save_path + f"critic_ckpt_{args.ckpt}.pt")
   170:         critic.load_state_dict(critic_ckpt["critic"])
   171:         critic_target.load_state_dict(critic_ckpt["critic_target"])
   172: 
   173:         actor.eval()
   174:         critic.eval()
   175:         critic_target.eval()
   176: 
   177:         env_eval = gym.vector.make(args.task.env_name, args.num_envs)
   178:         normalizer = dataset.get_normalizer()
   179:         episode_rewards = []
   180: 
   181:         # ============================================================================
   182:         # FIXED: NFE accounting — do not modify
   183:         # ============================================================================
   184:         # Counts real denoiser evaluations with a forward hook, so the reported
   185:         # NFE is what your sampler actually spends rather than a number declared
   186:         # in a config file. One hook call == one network evaluation, whatever
   187:         # batch it carries.
   188:         _nfe = {"calls": 0, "samples": 0}
   189: 
   190:         def _count_nfe(_module, _inputs, _output):
   191:             _nfe["calls"] += 1
   192: 
   193:         for _m in (actor.model, actor.model_ema):
   194:             try:
   195:                 _m["diffusion"].register_forward_hook(_count_nfe)
   196:             except (KeyError, TypeError, AttributeError):
   197:                 pass
   198: 
   199:         prior = torch.zeros((args.num_envs * args.num_candidates, act_dim), device=args.device)
   200:         for i in range(args.num_episodes):
   201: 
   202:             env_eval.seed(args.seed + i * args.num_envs) if hasattr(env_eval, "seed") else None; obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0
   203: 
   204:             while not np.all(cum_done) and t < 1000 + 1:
   205:                 obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
   206:                 obs = obs.unsqueeze(1).repeat(1, args.num_candidates, 1).view(-1, obs_dim)
   207: 
   208:                 _nfe["samples"] += 1
   209: 
   210:                 # ====================================================================
   211:                 # EDITABLE REGION: Sampling Algorithm
   212:                 # ====================================================================
   213:                 # Produce `act` of shape [num_envs * num_candidates, act_dim] by
   214:                 # running a reverse diffusion process conditioned on `obs`.
   215:                 #
   216:                 # The default below delegates to CleanDiffuser's built-in solvers,
   217:                 # driven by `solver` / `sampling_steps` in the YAML. You are not
   218:                 # limited to that: implement the reverse process yourself and call
   219:                 # the denoiser directly —
   220:                 #
   221:                 #   net = actor.model_ema["diffusion"] if args.use_ema else actor.model["diffusion"]
   222:                 #   pred = net(x_t, t, cond)        # eps or x0, per actor.predict_noise
   223:                 #
   224:                 # with the schedule available from actor.alphas / actor.sigmas (see
   225:                 # cleandiffuser/diffusion/diffusionsde.py). Fewer evaluations at the
   226:                 # same return is the point: every call to the denoiser is counted and
   227:                 # reported as the NFE the score penalizes, so the cost you pay is the
   228:                 # cost you are scored on.
   229:                 act, log = actor.sample(
   230:                     prior,
   231:                     solver=args.solver,
   232:                     n_samples=args.num_envs * args.num_candidates,
   233:                     sample_steps=args.sampling_steps,
   234:                     condition_cfg=obs, w_cfg=1.0,
   235:                     use_ema=args.use_ema, temperature=args.temperature)
   236:                 # ====================================================================
   237:                 # FIXED: Candidate Selection and Environment Step
   238:                 # ====================================================================
   239: 
   240:                 with torch.no_grad():
   241:                     q = critic_target.q_min(obs, act)
   242:                     q = q.view(-1, args.num_candidates, 1)
   243:                     w = torch.softmax(q * args.task.weight_temperature, 1)
   244:                     act = act.view(-1, args.num_candidates, act_dim)
   245: 
   246:                     indices = torch.multinomial(w.squeeze(-1), 1).squeeze(-1)
   247:                     sampled_act = act[torch.arange(act.shape[0]), indices].cpu().numpy()
   248: 
   249:                 obs, rew, done, info = env_eval.step(sampled_act)
   250: 
   251:                 t += 1
   252:                 cum_done = done if cum_done is None else np.logical_or(cum_done, done)
   253:                 ep_reward += (rew * (1 - cum_done)) if t < 1000 else rew
   254: 
   255:                 if np.all(cum_done):
   256:                     break
   257: 
   258:             episode_rewards.append(ep_reward)
   259: 
   260:         raw_episode_rewards = episode_rewards
   261:         episode_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
   262:         episode_rewards = np.array(episode_rewards)
   263:         mean_score = float(np.mean(episode_rewards))
   264:         std_score = float(np.std(episode_rewards))
   265:         mean_ep_reward = float(np.mean(raw_episode_rewards))
   266:         print(f"EVAL_METRICS normalized_score={mean_score:.4f} normalized_score_std={std_score:.4f} episode_reward={mean_ep_reward:.2f}")
   267: 
   268:         # Ceil, not round: an adaptive sampler averaging 10.5 evaluations must
   269:         # not report 10 and collect the full no-penalty credit reserved for a
   270:         # 10-step budget. Round-half-to-even would also floor an average of 0.5
   271:         # to zero. Constant-call samplers are unaffected.
   272:         measured_nfe = -(-_nfe["calls"] // max(_nfe["samples"], 1))
   273:         print(f"NFE_METRICS sampling_steps={measured_nfe}", flush=True)
   274: 
   275:     else:
   276:         raise ValueError(f"Invalid mode: {args.mode}")
   277: 
   278: 
   279: if __name__ == "__main__":
   280:     pipeline()
```

### `CleanDiffuser/configs/custom/mujoco/mujoco.yaml`  [EDITABLE — lines 15–15, lines 17–17 only]

```yaml
     1: defaults:
     2:   - _self_
     3:   - task: hopper-medium-v2
     4: 
     5: pipeline_name: custom_sampling_method
     6: mode: train
     7: seed: 42
     8: device: cuda:0
     9: 
    10: # Environment
    11: normalize_reward: True
    12: discount: 0.99
    13: 
    14: # Actor
    15: solver: ddpm
    16: diffusion_steps: 100
    17: sampling_steps: 100
    18: predict_noise: True
    19: ema_rate: 0.995
    20: actor_learning_rate: 0.0003
    21: 
    22: # Critic
    23: hidden_dim: 256
    24: critic_learning_rate: 0.0003
    25: 
    26: # Training
    27: gradient_steps: 100000
    28: batch_size: 256
    29: ema_update_interval: 5
    30: log_interval: 1000
    31: save_interval: 50000
    32: 
    33: # Inference
    34: ckpt: latest
    35: num_envs: 50
    36: num_episodes: 3
    37: num_candidates: 50
    38: temperature: 0.5
    39: use_ema: True
    40: 
    41: # hydra
    42: hydra:
    43:   job:
    44:     chdir: false
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


### `ddim` baseline — editable region  [READ-ONLY — reference implementation]

In `CleanDiffuser/configs/custom/mujoco/mujoco.yaml`:

```python
Lines 15–15:
    12: discount: 0.99
    13: 
    14: # Actor
    15: solver: ddim
    16: diffusion_steps: 100
    17: sampling_steps: 20
    18: predict_noise: True

Lines 17–17:
    14: # Actor
    15: solver: ddim
    16: diffusion_steps: 100
    17: sampling_steps: 20
    18: predict_noise: True
    19: ema_rate: 0.995
    20: actor_learning_rate: 0.0003
```

### `dpm_solver` baseline — editable region  [READ-ONLY — reference implementation]

In `CleanDiffuser/configs/custom/mujoco/mujoco.yaml`:

```python
Lines 15–15:
    12: discount: 0.99
    13: 
    14: # Actor
    15: solver: ode_dpmsolver++_2M
    16: diffusion_steps: 100
    17: sampling_steps: 10
    18: predict_noise: True

Lines 17–17:
    14: # Actor
    15: solver: ode_dpmsolver++_2M
    16: diffusion_steps: 100
    17: sampling_steps: 10
    18: predict_noise: True
    19: ema_rate: 0.995
    20: actor_learning_rate: 0.0003
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

## Time Budget

You have **5 hours** of wall-clock time before submission, covering
everything you do here: reading the code, editing it, and any trial runs
you launch.

Good luck.
