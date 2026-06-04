# MLS-Bench: robo-diffusion-guidance

# Robo-Diffusion: Guided Sampling Strategy Design

## Objective
Design one improved guidance mechanism for a fixed trajectory-level diffusion planner on offline locomotion benchmarks. This task is narrower than `robo-diffusion-policy`: the research question is how to condition or guide the reverse diffusion process, not how to redesign the whole planner or the model-free actor-critic training loop.

## Background
Diffusion-based decision-making models support two main guidance paradigms:
1. **Classifier Guidance (CG)**: a separately trained classifier (here a cumulative-reward predictor) injects gradients into the reverse process. References: Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis", NeurIPS 2021 (arXiv:2105.05233); Janner et al., Diffuser, ICML 2022 (arXiv:2205.09991).
2. **Classifier-Free Guidance (CFG)**: the diffusion network is jointly trained with and without a condition (return-to-go) and at sample time the conditional and unconditional predictions are interpolated by `w_cfg`. References: Ho & Salimans, "Classifier-Free Diffusion Guidance" (arXiv:2207.12598); Ajay et al., Decision Diffuser, ICLR 2023 (arXiv:2211.15657).

The choice of guidance strategy and its weight schedule strongly affects both final reward and inference cost.

The implementation builds on **CleanDiffuser** (Dong et al., NeurIPS 2024, arXiv:2406.09509), a modular diffusion library for decision making.

## What Is Implemented (the starting code you edit)
The editable file is **`CleanDiffuser/pipelines/custom_guidance.py`**, which is created by `mid_edit.py` from `edits/custom_template.py`. Concretely:
- **Backbone**: `JannerUNet1d` 1-D U-Net diffusing trajectories of dimension `obs_dim + act_dim` (state + action), with kernel size 5, no attention.
- **Diffusion process**: `DiscreteDiffusionSDE` with `diffusion_steps=20`, `predict_noise=False`, `solver=ddpm`, `ema_rate=0.9999`.
- **Default guidance**: classifier guidance via `CumRewClassifier` (`HalfJannerUNet1d` head). At sample time, `args.task.w_cg` is used and `num_candidates=64` trajectories are drawn per env-step, then re-ranked by the classifier's log-probability.
- **Conditioning entry-points already wired by the config**: each task yaml exposes `w_cg`, `w_cfg`, and `target_return`. `w_cg` is used by the default (CG) path; `w_cfg` and `target_return` are consumed by CFG / Decision Diffuser variants. You may use any of them.

There is no `apply_guidance` standalone function — guidance is configured by the choice of `nn_condition` / `classifier`, the `agent.update(...)` call during training, and the `agent.sample(..., w_cg=, w_cfg=, condition_cfg=)` call during inference.

## What You Can Modify
You can modify any of the following inside the editable regions of `custom_guidance.py`:
- Network architecture for the diffusion model (must remain compatible with `DiscreteDiffusionSDE` / `ContinuousDiffusionSDE`).
- Classifier architecture and training objective (or remove it).
- Condition network (e.g. swap or augment `MLPCondition`).
- Training loop: how `agent.update(...)` is called, label dropout, return normalization, mixed CG+CFG schedules.
- Sampling loop: `w_cg`, `w_cfg`, time-dependent guidance weights, candidate re-ranking strategy, EMA / sample steps, etc.
- New guidance strategies: hybrid CG+CFG, adaptive `w_cfg(t)`, late-only guidance, gradient-clipping schedules, novel classifier targets.

## What Is Fixed
- Dataset and environment loop (`env.get_dataset()`, `gym.vector.make`, normalization, reward collection).
- Environment names, seeds, and reward computation.
- Top-level training hyperparameters in `mujoco.yaml`: `diffusion_gradient_steps=100000`, `batch_size=256`, `model_dim=32`, `solver=ddpm`, `diffusion_steps=20`, `sampling_steps=20`.

## Baselines
Four baselines are provided in `edits/`:

### 1. `default` — Diffuser (Classifier Guidance, CG)
- Unmodified template. `JannerUNet1d` + `CumRewClassifier`, guidance weight `w_cg` set per environment.
- Inference uses 64 candidates re-ranked by classifier log-prob.
- Reference: Janner et al., 2022, Diffuser, arXiv:2205.09991.

### 2. `cfg` — minimal CFG ablation of the default
- **Same** `JannerUNet1d` backbone, **same** `DiscreteDiffusionSDE`, **same** obs+act trajectory diffusion as `default`.
- Replaces `CumRewClassifier` with an `MLPCondition` over normalized return, trained with label dropout. Sampling uses `condition_cfg=target_return`, `w_cfg` set per environment (Decision Diffuser paper values), `w_cg = 0`. No candidate re-ranking.

### 3. `no_guidance` — unconditional ablation
- `JannerUNet1d` trained without any classifier or condition network.
- Sampling with `w_cg = w_cfg = 0`, single sample per env-step (no re-ranking).

### 4. `decision_diffuser` — full CFG with DD architecture
- Verbatim port of CleanDiffuser's `dd_d4rl_mujoco.py` (Ajay et al., 2022, arXiv:2211.15657).
- State-only diffusion (`obs` rather than `obs+act`), DiT1d Transformer backbone, `MlpInvDynamic` to recover actions, `ContinuousDiffusionSDE`.

## Tips
- Guidance strength matters most in late-time steps (small `t`). Time-dependent `w_cfg(t)` schedules are an underexplored axis.
- Combining CG and CFG additively in `agent.sample` (`w_cg > 0` AND `w_cfg > 0`) is supported by `DiscreteDiffusionSDE`.
- Re-ranking by a learned value head is a major source of CG performance — consider whether your method needs or replaces it.

## Files You Edit
- `CleanDiffuser/pipelines/custom_guidance.py` — created from `edits/custom_template.py`. Editable regions cover network setup, training loop, inference setup, prior / condition initialization, and action sampling.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/CleanDiffuser/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `CleanDiffuser/pipelines/custom_guidance.py`
- editable lines **1–28**
- editable lines **38–136**
- editable lines **145–152**
- editable lines **164–183**




## Readable Context


### `CleanDiffuser/pipelines/custom_guidance.py`  [EDITABLE — lines 1–28, lines 38–136, lines 145–152, lines 164–183 only]

```python
     1: import os
     2: 
     3: import d4rl
     4: import gym
     5: import hydra
     6: import numpy as np
     7: import torch
     8: from torch.optim.lr_scheduler import CosineAnnealingLR
     9: from torch.utils.data import DataLoader
    10: 
    11: from cleandiffuser.classifier import CumRewClassifier
    12: from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoDataset
    13: from cleandiffuser.dataset.dataset_utils import loop_dataloader
    14: from cleandiffuser.diffusion import DiscreteDiffusionSDE
    15: from cleandiffuser.nn_classifier import HalfJannerUNet1d
    16: from cleandiffuser.nn_diffusion import JannerUNet1d
    17: from cleandiffuser.utils import report_parameters
    18: from utils import set_seed
    19: 
    20: 
    21: @hydra.main(config_path="../configs/custom/mujoco", config_name="mujoco", version_base=None)
    22: def pipeline(args):
    23: 
    24:     set_seed(args.seed)
    25: 
    26:     save_path = f'results/{args.pipeline_name}/{args.task.env_name}/'
    27:     if os.path.exists(save_path) is False:
    28:         os.makedirs(save_path)
    29: 
    30:     # ---------------------- Create Dataset ----------------------
    31:     env = gym.make(args.task.env_name)
    32:     dataset = D4RLMuJoCoDataset(
    33:         env.get_dataset(), horizon=args.task.horizon, terminal_penalty=args.terminal_penalty, discount=args.discount)
    34:     dataloader = DataLoader(
    35:         dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    36:     obs_dim, act_dim = dataset.o_dim, dataset.a_dim
    37: 
    38:     # ============================================================================
    39:     # EDITABLE REGION 3: Network + Agent Setup (lines 40-72)
    40:     # ============================================================================
    41: 
    42:     # --------------- Network Architecture -----------------
    43:     nn_diffusion = JannerUNet1d(
    44:         obs_dim + act_dim, model_dim=args.model_dim, emb_dim=args.model_dim, dim_mult=args.task.dim_mult,
    45:         timestep_emb_type="positional", attention=False, kernel_size=5)
    46:     nn_classifier = HalfJannerUNet1d(
    47:         args.task.horizon, obs_dim + act_dim, out_dim=1,
    48:         model_dim=args.model_dim, emb_dim=args.model_dim, dim_mult=args.task.dim_mult,
    49:         timestep_emb_type="positional", kernel_size=3)
    50: 
    51:     print(f"======================= Parameter Report of Diffusion Model =======================")
    52:     report_parameters(nn_diffusion)
    53:     print(f"======================= Parameter Report of Classifier =======================")
    54:     report_parameters(nn_classifier)
    55:     print(f"==============================================================================")
    56: 
    57:     # --------------- Classifier Guidance --------------------
    58:     classifier = CumRewClassifier(nn_classifier, device=args.device)
    59: 
    60:     # ----------------- Masking -------------------
    61:     fix_mask = torch.zeros((args.task.horizon, obs_dim + act_dim))
    62:     fix_mask[0, :obs_dim] = 1.
    63:     loss_weight = torch.ones((args.task.horizon, obs_dim + act_dim))
    64:     loss_weight[0, obs_dim:] = args.action_loss_weight
    65: 
    66:     # --------------- Diffusion Model --------------------
    67:     agent = DiscreteDiffusionSDE(
    68:         nn_diffusion, None,
    69:         fix_mask=fix_mask, loss_weight=loss_weight, classifier=classifier, ema_rate=args.ema_rate,
    70:         device=args.device, diffusion_steps=args.diffusion_steps, predict_noise=args.predict_noise)
    71: 
    72:     # ============================================================================
    73:     # EDITABLE REGION 4: Training + Finetune (lines 74-182)
    74:     # ============================================================================
    75: 
    76:     # ---------------------- Training ----------------------
    77:     if args.mode == "train":
    78: 
    79:         diffusion_lr_scheduler = CosineAnnealingLR(agent.optimizer, args.diffusion_gradient_steps)
    80:         classifier_lr_scheduler = CosineAnnealingLR(agent.classifier.optim, args.classifier_gradient_steps)
    81: 
    82:         agent.train()
    83: 
    84:         n_gradient_step = 0
    85:         log = {"avg_loss_diffusion": 0., "avg_loss_classifier": 0.}
    86: 
    87:         for batch in loop_dataloader(dataloader):
    88: 
    89:             obs = batch["obs"]["state"].to(args.device)
    90:             act = batch["act"].to(args.device)
    91:             val = batch["val"].to(args.device)
    92: 
    93:             x = torch.cat([obs, act], -1)
    94: 
    95:             # ----------- Gradient Step ------------
    96:             log["avg_loss_diffusion"] += agent.update(x)['loss']
    97:             diffusion_lr_scheduler.step()
    98:             if n_gradient_step <= args.classifier_gradient_steps:
    99:                 log["avg_loss_classifier"] += agent.update_classifier(x, val)['loss']
   100:                 classifier_lr_scheduler.step()
   101: 
   102:             # ----------- Logging ------------
   103:             if (n_gradient_step + 1) % args.log_interval == 0:
   104:                 log["gradient_steps"] = n_gradient_step + 1
   105:                 log["avg_loss_diffusion"] /= args.log_interval
   106:                 log["avg_loss_classifier"] /= args.log_interval
   107:                 print(log)
   108:                 log = {"avg_loss_diffusion": 0., "avg_loss_classifier": 0.}
   109: 
   110:             # ----------- Saving ------------
   111:             if (n_gradient_step + 1) % args.save_interval == 0:
   112:                 agent.save(save_path + f"diffusion_ckpt_{n_gradient_step + 1}.pt")
   113:                 agent.classifier.save(save_path + f"classifier_ckpt_{n_gradient_step + 1}.pt")
   114:                 agent.save(save_path + f"diffusion_ckpt_latest.pt")
   115:                 agent.classifier.save(save_path + f"classifier_ckpt_latest.pt")
   116: 
   117:             n_gradient_step += 1
   118:             if n_gradient_step >= args.diffusion_gradient_steps:
   119:                 break
   120: 
   121:     # ---------------------- Finetune (placeholder for adaptdiffuser) ----------------------
   122:     elif args.mode == "finetune":
   123:         pass
   124: 
   125:     # ---------------------- Inference ----------------------
   126:     elif args.mode == "inference":
   127: 
   128:         # ============================================================================
   129:         # EDITABLE REGION 5: Inference Setup (lines 186-197)
   130:         # ============================================================================
   131: 
   132:         agent.load(save_path + f"diffusion_ckpt_{args.ckpt}.pt")
   133:         agent.classifier.load(save_path + f"classifier_ckpt_{args.ckpt}.pt")
   134: 
   135:         agent.eval()
   136: 
   137:         # ============================================================================
   138:         # FIXED: Environment Setup (lines 198-206)
   139:         # ============================================================================
   140: 
   141:         env_eval = gym.vector.make(args.task.env_name, args.num_envs)
   142:         normalizer = dataset.get_normalizer()
   143:         episode_rewards = []
   144: 
   145:         # ============================================================================
   146:         # EDITABLE REGION 6: Prior + Condition Initialization (lines 207-222)
   147:         # ============================================================================
   148: 
   149:         prior = torch.zeros((args.num_envs, args.task.horizon, obs_dim + act_dim), device=args.device)
   150: 
   151:         for i in range(args.num_episodes):
   152: 
   153:             env_eval.seed(args.seed + i * args.num_envs) if hasattr(env_eval, "seed") else None; obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0
   154: 
   155:             while not np.all(cum_done) and t < 1000 + 1:
   156: 
   157:                 # ============================================================================
   158:                 # FIXED: Observation Normalization (lines 223-225)
   159:                 # ============================================================================
   160: 
   161:                 # normalize obs
   162:                 obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
   163: 
   164:                 # ============================================================================
   165:                 # EDITABLE REGION 7: Action Sampling (lines 226-240)
   166:                 # ============================================================================
   167: 
   168:                 # sample trajectories
   169:                 prior[:, 0, :obs_dim] = obs
   170:                 traj, log = agent.sample(
   171:                     prior.repeat(args.num_candidates, 1, 1),
   172:                     solver=args.solver,
   173:                     n_samples=args.num_candidates * args.num_envs,
   174:                     sample_steps=args.sampling_steps,
   175:                     use_ema=args.use_ema, w_cg=args.task.w_cg, temperature=args.temperature)
   176: 
   177:                 # select the best plan
   178:                 logp = log["log_p"].view(args.num_candidates, args.num_envs, -1).sum(-1)
   179:                 idx = logp.argmax(0)
   180:                 act = traj.view(args.num_candidates, args.num_envs, args.task.horizon, -1)[
   181:                       idx, torch.arange(args.num_envs), 0, obs_dim:]
   182:                 act = act.clip(-1., 1.).cpu().numpy()
   183: 
   184:                 # ============================================================================
   185:                 # FIXED: Environment Step + Reward Collection (lines 241-252)
   186:                 # ============================================================================
   187: 
   188:                 # step
   189:                 obs, rew, done, info = env_eval.step(act)
   190: 
   191:                 t += 1
   192:                 cum_done = done if cum_done is None else np.logical_or(cum_done, done)
   193:                 ep_reward += (rew * (1 - cum_done)) if t < 1000 else rew
   194:                 print(f'[t={t}] rew: {np.around((rew * (1 - cum_done)), 2)}, '
   195:                       f'logp: {logp[idx, torch.arange(args.num_envs)]}')
   196: 
   197:             episode_rewards.append(ep_reward)
   198: 
   199:         # ============================================================================
   200:         # FIXED: Final Scoring (lines 253-257)
   201:         # ============================================================================
   202: 
   203:         raw_episode_rewards = episode_rewards
   204:         episode_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
   205:         episode_rewards = np.array(episode_rewards)
   206:         mean_score = float(np.mean(episode_rewards))
   207:         std_score = float(np.std(episode_rewards))
   208:         mean_ep_reward = float(np.mean(raw_episode_rewards))
   209:         print(f"EVAL_METRICS normalized_score={mean_score:.4f} normalized_score_std={std_score:.4f} episode_reward={mean_ep_reward:.2f}")
   210:         print(np.mean(episode_rewards, -1), np.std(episode_rewards, -1))
   211: 
   212:     else:
   213:         raise ValueError(f"Invalid mode: {args.mode}")
   214: 
   215: 
   216: if __name__ == "__main__":
   217:     pipeline()
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


### `cfg` baseline — editable region  [READ-ONLY — reference implementation]

In `CleanDiffuser/pipelines/custom_guidance.py`:

```python
Lines 1–28:
     1: import os
     2: 
     3: import d4rl
     4: import gym
     5: import hydra
     6: import numpy as np
     7: import torch
     8: import torch.nn as nn
     9: from torch.optim.lr_scheduler import CosineAnnealingLR
    10: from torch.utils.data import DataLoader
    11: 
    12: from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoDataset
    13: from cleandiffuser.dataset.dataset_utils import loop_dataloader
    14: from cleandiffuser.diffusion import DiscreteDiffusionSDE
    15: from cleandiffuser.nn_condition import MLPCondition
    16: from cleandiffuser.nn_diffusion import JannerUNet1d
    17: from cleandiffuser.utils import report_parameters, DD_RETURN_SCALE
    18: from utils import set_seed
    19: 
    20: 
    21: @hydra.main(config_path="../configs/custom/mujoco", config_name="mujoco", version_base=None)
    22: def pipeline(args):
    23: 
    24:     set_seed(args.seed)
    25: 
    26:     save_path = f'results/{args.pipeline_name}/{args.task.env_name}/'
    27:     if os.path.exists(save_path) is False:
    28:         os.makedirs(save_path)
    29: 
    30:     # ---------------------- Create Dataset ----------------------
    31:     env = gym.make(args.task.env_name)

Lines 38–130:
    35:         dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    36:     obs_dim, act_dim = dataset.o_dim, dataset.a_dim
    37: 
    38:     # ============================================================================
    39:     # cfg: Network + Agent Setup (JannerUNet1d + MLPCondition, no classifier)
    40:     # ============================================================================
    41: 
    42:     return_scale = DD_RETURN_SCALE[args.task.env_name]
    43: 
    44:     # --------------- Network Architecture (same as default) -----------------
    45:     nn_diffusion = JannerUNet1d(
    46:         obs_dim + act_dim, model_dim=args.model_dim, emb_dim=args.model_dim, dim_mult=args.task.dim_mult,
    47:         timestep_emb_type="positional", attention=False, kernel_size=5)
    48: 
    49:     # --------------- Classifier-Free condition network (replaces classifier) -----------------
    50:     nn_condition = MLPCondition(
    51:         in_dim=1, out_dim=args.model_dim,
    52:         hidden_dims=[args.model_dim, ], act=nn.SiLU(), dropout=args.label_dropout)
    53: 
    54:     print(f"======================= Parameter Report of Diffusion Model =======================")
    55:     report_parameters(nn_diffusion)
    56:     # report_parameters(nn_condition) crashes when nn_condition has fewer params than its
    57:     # hardcoded top-K (sorted_keys[i] out of range for the tiny MLPCondition). Skip and
    58:     # just print the total so logs stay informative.
    59:     print(f"======================= Condition Network: MLPCondition =======================")
    60:     print(f"Total parameters: {sum(p.numel() for p in nn_condition.parameters())}")
    61:     print(f"==============================================================================")
    62: 
    63:     # ----------------- Masking (identical to default) -------------------
    64:     fix_mask = torch.zeros((args.task.horizon, obs_dim + act_dim))
    65:     fix_mask[0, :obs_dim] = 1.
    66:     loss_weight = torch.ones((args.task.horizon, obs_dim + act_dim))
    67:     loss_weight[0, obs_dim:] = args.action_loss_weight
    68: 
    69:     # --------------- Diffusion Model (same SDE, classifier=None, condition=MLP) --------------------
    70:     agent = DiscreteDiffusionSDE(
    71:         nn_diffusion, nn_condition,
    72:         fix_mask=fix_mask, loss_weight=loss_weight, classifier=None, ema_rate=args.ema_rate,
    73:         device=args.device, diffusion_steps=args.diffusion_steps, predict_noise=args.predict_noise)
    74:     # ============================================================================
    75:     # cfg: Training (diffusion only with return-conditioning, no classifier)
    76:     # ============================================================================
    77: 
    78:     # ---------------------- Training ----------------------
    79:     if args.mode == "train":
    80: 
    81:         diffusion_lr_scheduler = CosineAnnealingLR(agent.optimizer, args.diffusion_gradient_steps)
    82: 
    83:         agent.train()
    84: 
    85:         n_gradient_step = 0
    86:         log = {"avg_loss_diffusion": 0.}
    87: 
    88:         for batch in loop_dataloader(dataloader):
    89: 
    90:             obs = batch["obs"]["state"].to(args.device)
    91:             act = batch["act"].to(args.device)
    92:             val = batch["val"].to(args.device) / return_scale
    93: 
    94:             x = torch.cat([obs, act], -1)
    95: 
    96:             # ----------- Gradient Step (CFG-style: condition with label dropout) ------------
    97:             log["avg_loss_diffusion"] += agent.update(x, val)['loss']
    98:             diffusion_lr_scheduler.step()
    99: 
   100:             # ----------- Logging ------------
   101:             if (n_gradient_step + 1) % args.log_interval == 0:
   102:                 log["gradient_steps"] = n_gradient_step + 1
   103:                 log["avg_loss_diffusion"] /= args.log_interval
   104:                 print(log)
   105:                 log = {"avg_loss_diffusion": 0.}
   106: 
   107:             # ----------- Saving ------------
   108:             if (n_gradient_step + 1) % args.save_interval == 0:
   109:                 agent.save(save_path + f"diffusion_ckpt_{n_gradient_step + 1}.pt")
   110:                 agent.save(save_path + f"diffusion_ckpt_latest.pt")
   111: 
   112:             n_gradient_step += 1
   113:             if n_gradient_step >= args.diffusion_gradient_steps:
   114:                 break
   115: 
   116:     # ---------------------- Finetune (placeholder) ----------------------
   117:     elif args.mode == "finetune":
   118:         pass
   119: 
   120:     # ---------------------- Inference ----------------------
   121:     elif args.mode == "inference":
   122: 
   123:         # ============================================================================
   124:         # cfg: Inference Setup (diffusion only)
   125:         # ============================================================================
   126: 
   127:         agent.load(save_path + f"diffusion_ckpt_{args.ckpt}.pt")
   128: 
   129:         agent.eval()
   130: 
   131:         # ============================================================================
   132:         # FIXED: Environment Setup (lines 198-206)
   133:         # ============================================================================

Lines 139–146:
   136:         normalizer = dataset.get_normalizer()
   137:         episode_rewards = []
   138: 
   139:         # ============================================================================
   140:         # cfg: Prior + Condition Initialization (return-conditioned via MLPCondition)
   141:         # ============================================================================
   142: 
   143:         prior = torch.zeros((args.num_envs, args.task.horizon, obs_dim + act_dim), device=args.device)
   144:         condition = torch.ones((args.num_envs, 1), device=args.device) * args.task.target_return
   145: 
   146:         for i in range(args.num_episodes):
   147:             env_eval.seed(args.seed + i * args.num_envs) if hasattr(env_eval, "seed") else None; obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0
   148: 
   149:             while not np.all(cum_done) and t < 1000 + 1:

Lines 158–183:
   155:                 # normalize obs
   156:                 obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
   157: 
   158:                 # ============================================================================
   159:                 # cfg: Action Sampling (CFG, no candidate re-ranking)
   160:                 # ============================================================================
   161: 
   162:                 # sample trajectories conditioned on target return
   163:                 prior[:, 0, :obs_dim] = obs
   164:                 traj, log = agent.sample(
   165:                     prior,
   166:                     solver=args.solver,
   167:                     n_samples=args.num_envs,
   168:                     sample_steps=args.sampling_steps,
   169:                     use_ema=args.use_ema,
   170:                     condition_cfg=condition,
   171:                     w_cfg=args.task.w_cfg,
   172:                     w_cg=0.0,
   173:                     temperature=args.temperature)
   174: 
   175:                 # read actions directly from the conditional sample (no logp re-ranking)
   176:                 act = traj[:, 0, obs_dim:]
   177:                 act = act.clip(-1., 1.).cpu().numpy()
   178: 
   179:                 # FIXED post-sample print (outside editable range) references logp/idx;
   180:                 # CFG has no candidate re-ranking, so define harmless placeholders.
   181:                 logp = torch.zeros((1, args.num_envs), device=args.device)
   182:                 idx = torch.zeros((args.num_envs,), dtype=torch.long, device=args.device)
   183: 
   184:                 # ============================================================================
   185:                 # FIXED: Environment Step + Reward Collection (lines 241-252)
   186:                 # ============================================================================
```

### `no_guidance` baseline — editable region  [READ-ONLY — reference implementation]

In `CleanDiffuser/pipelines/custom_guidance.py`:

```python
Lines 1–26:
     1: import os
     2: 
     3: import d4rl
     4: import gym
     5: import hydra
     6: import numpy as np
     7: import torch
     8: from torch.optim.lr_scheduler import CosineAnnealingLR
     9: from torch.utils.data import DataLoader
    10: 
    11: from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoDataset
    12: from cleandiffuser.dataset.dataset_utils import loop_dataloader
    13: from cleandiffuser.diffusion import DiscreteDiffusionSDE
    14: from cleandiffuser.nn_diffusion import JannerUNet1d
    15: from cleandiffuser.utils import report_parameters
    16: from utils import set_seed
    17: 
    18: 
    19: @hydra.main(config_path="../configs/custom/mujoco", config_name="mujoco", version_base=None)
    20: def pipeline(args):
    21: 
    22:     set_seed(args.seed)
    23: 
    24:     save_path = f'results/{args.pipeline_name}/{args.task.env_name}/'
    25:     if os.path.exists(save_path) is False:
    26:         os.makedirs(save_path)
    27: 
    28:     # ---------------------- Create Dataset ----------------------
    29:     env = gym.make(args.task.env_name)

Lines 36–110:
    33:         dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    34:     obs_dim, act_dim = dataset.o_dim, dataset.a_dim
    35: 
    36:     # ============================================================================
    37:     # no_guidance: Network + Agent Setup (no classifier)
    38:     # ============================================================================
    39: 
    40:     nn_diffusion = JannerUNet1d(
    41:         obs_dim + act_dim, model_dim=args.model_dim, emb_dim=args.model_dim, dim_mult=args.task.dim_mult,
    42:         timestep_emb_type="positional", attention=False, kernel_size=5)
    43: 
    44:     print(f"======================= Parameter Report of Diffusion Model =======================")
    45:     report_parameters(nn_diffusion)
    46:     print(f"==============================================================================")
    47: 
    48:     fix_mask = torch.zeros((args.task.horizon, obs_dim + act_dim))
    49:     fix_mask[0, :obs_dim] = 1.
    50:     loss_weight = torch.ones((args.task.horizon, obs_dim + act_dim))
    51:     loss_weight[0, obs_dim:] = args.action_loss_weight
    52: 
    53:     agent = DiscreteDiffusionSDE(
    54:         nn_diffusion, None,
    55:         fix_mask=fix_mask, loss_weight=loss_weight, classifier=None, ema_rate=args.ema_rate,
    56:         device=args.device, diffusion_steps=args.diffusion_steps, predict_noise=args.predict_noise)
    57: 
    58:     # ============================================================================
    59:     # no_guidance: Training (diffusion only, no classifier)
    60:     # ============================================================================
    61: 
    62:     if args.mode == "train":
    63: 
    64:         diffusion_lr_scheduler = CosineAnnealingLR(agent.optimizer, args.diffusion_gradient_steps)
    65: 
    66:         agent.train()
    67: 
    68:         n_gradient_step = 0
    69:         log = {"avg_loss_diffusion": 0.}
    70: 
    71:         for batch in loop_dataloader(dataloader):
    72: 
    73:             obs = batch["obs"]["state"].to(args.device)
    74:             act = batch["act"].to(args.device)
    75: 
    76:             x = torch.cat([obs, act], -1)
    77: 
    78:             log["avg_loss_diffusion"] += agent.update(x)['loss']
    79:             diffusion_lr_scheduler.step()
    80: 
    81:             if (n_gradient_step + 1) % args.log_interval == 0:
    82:                 log["gradient_steps"] = n_gradient_step + 1
    83:                 log["avg_loss_diffusion"] /= args.log_interval
    84:                 print(log)
    85:                 log = {"avg_loss_diffusion": 0.}
    86: 
    87:             if (n_gradient_step + 1) % args.save_interval == 0:
    88:                 agent.save(save_path + f"diffusion_ckpt_{n_gradient_step + 1}.pt")
    89:                 agent.save(save_path + f"diffusion_ckpt_latest.pt")
    90: 
    91:             n_gradient_step += 1
    92:             if n_gradient_step >= args.diffusion_gradient_steps:
    93:                 break
    94: 
    95:     elif args.mode == "finetune":
    96:         pass
    97: 
    98: 
    99:     # ---------------------- Inference ----------------------
   100:     elif args.mode == "inference":
   101: 
   102:         # ============================================================================
   103:         # no_guidance: Inference Setup (diffusion only)
   104:         # ============================================================================
   105: 
   106:         agent.load(save_path + f"diffusion_ckpt_{args.ckpt}.pt")
   107: 
   108:         agent.eval()
   109: 
   110: 
   111:         # ============================================================================
   112:         # FIXED: Environment Setup (lines 198-206)
   113:         # ============================================================================

Lines 119–125:
   116:         normalizer = dataset.get_normalizer()
   117:         episode_rewards = []
   118: 
   119:         # ============================================================================
   120:         # no_guidance: Prior Initialization (no condition, no target return)
   121:         # ============================================================================
   122: 
   123:         prior = torch.zeros((args.num_envs, args.task.horizon, obs_dim + act_dim), device=args.device)
   124: 
   125:         for i in range(args.num_episodes):
   126:             env_eval.seed(args.seed + i * args.num_envs) if hasattr(env_eval, "seed") else None; obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0
   127: 
   128:             while not np.all(cum_done) and t < 1000 + 1:

Lines 137–159:
   134:                 # normalize obs
   135:                 obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
   136: 
   137:                 # ============================================================================
   138:                 # no_guidance: Action Sampling (w_cg=0, no re-ranking)
   139:                 # ============================================================================
   140: 
   141:                 prior[:, 0, :obs_dim] = obs
   142:                 traj, log = agent.sample(
   143:                     prior,
   144:                     solver=args.solver,
   145:                     n_samples=args.num_envs,
   146:                     sample_steps=args.sampling_steps,
   147:                     use_ema=args.use_ema,
   148:                     w_cg=0.0,
   149:                     w_cfg=0.0,
   150:                     temperature=args.temperature)
   151: 
   152:                 act = traj[:, 0, obs_dim:]
   153:                 act = act.clip(-1., 1.).cpu().numpy()
   154: 
   155:                 # FIXED post-sample print (outside editable range) references logp/idx;
   156:                 # no_guidance has no candidate re-ranking, so define harmless placeholders.
   157:                 logp = torch.zeros((1, args.num_envs), device=args.device)
   158:                 idx = torch.zeros((args.num_envs,), dtype=torch.long, device=args.device)
   159: 
   160:                 # ============================================================================
   161:                 # FIXED: Environment Step + Reward Collection (lines 241-252)
   162:                 # ============================================================================
```

### `decision_diffuser` baseline — editable region  [READ-ONLY — reference implementation]

In `CleanDiffuser/pipelines/custom_guidance.py`:

```python
Lines 1–31:
     1: import os
     2: 
     3: import d4rl
     4: import gym
     5: import hydra
     6: import numpy as np
     7: import torch
     8: import torch.nn as nn
     9: from torch.optim.lr_scheduler import CosineAnnealingLR
    10: from torch.utils.data import DataLoader
    11: 
    12: from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoDataset
    13: from cleandiffuser.dataset.dataset_utils import loop_dataloader
    14: from cleandiffuser.diffusion import ContinuousDiffusionSDE
    15: from cleandiffuser.invdynamic import MlpInvDynamic
    16: from cleandiffuser.nn_condition import MLPCondition
    17: from cleandiffuser.nn_diffusion import DiT1d
    18: from cleandiffuser.utils import report_parameters, DD_RETURN_SCALE
    19: from utils import set_seed
    20: 
    21: 
    22: @hydra.main(config_path="../configs/dd/mujoco", config_name="mujoco", version_base=None)
    23: def pipeline(args):
    24: 
    25:     return_scale = DD_RETURN_SCALE[args.task.env_name]
    26: 
    27:     set_seed(args.seed)
    28: 
    29:     save_path = f'results/{args.pipeline_name}/{args.task.env_name}/'
    30:     if os.path.exists(save_path) is False:
    31:         os.makedirs(save_path)
    32: 
    33:     # ---------------------- Create Dataset ----------------------
    34:     env = gym.make(args.task.env_name)

Lines 41–133:
    38:         dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    39:     obs_dim, act_dim = dataset.o_dim, dataset.a_dim
    40: 
    41:     # ============================================================================
    42:     # EDITABLE REGION 3: Network + Agent Setup (lines 40-72)
    43:     # ============================================================================
    44: 
    45:     # --------------- Network Architecture -----------------
    46:     nn_diffusion = DiT1d(
    47:         obs_dim, emb_dim=args.emb_dim,
    48:         d_model=args.d_model, n_heads=args.n_heads, depth=args.depth, timestep_emb_type="fourier")
    49:     nn_condition = MLPCondition(
    50:         in_dim=1, out_dim=args.emb_dim, hidden_dims=[args.emb_dim, ], act=nn.SiLU(), dropout=args.label_dropout)
    51: 
    52:     print(f"======================= Parameter Report of Diffusion Model =======================")
    53:     report_parameters(nn_diffusion)
    54:     print(f"==============================================================================")
    55: 
    56:     # ----------------- Masking -------------------
    57:     fix_mask = torch.zeros((args.task.horizon, obs_dim))
    58:     fix_mask[0] = 1.
    59:     loss_weight = torch.ones((args.task.horizon, obs_dim))
    60:     loss_weight[1] = args.next_obs_loss_weight
    61: 
    62:     # --------------- Diffusion Model with Classifier-Free Guidance --------------------
    63:     agent = ContinuousDiffusionSDE(
    64:         nn_diffusion, nn_condition,
    65:         fix_mask=fix_mask, loss_weight=loss_weight, ema_rate=args.ema_rate,
    66:         device=args.device, predict_noise=args.predict_noise, noise_schedule="linear")
    67: 
    68:     # --------------- Inverse Dynamic -------------------
    69:     invdyn = MlpInvDynamic(obs_dim, act_dim, 512, nn.Tanh(), {"lr": 2e-4}, device=args.device)
    70:     # ============================================================================
    71:     # EDITABLE REGION 4: Training + Finetune (lines 74-182)
    72:     # ============================================================================
    73: 
    74:     # ---------------------- Training ----------------------
    75:     if args.mode == "train":
    76: 
    77:         diffusion_lr_scheduler = CosineAnnealingLR(agent.optimizer, args.diffusion_gradient_steps)
    78:         invdyn_lr_scheduler = CosineAnnealingLR(invdyn.optim, args.invdyn_gradient_steps)
    79: 
    80:         agent.train()
    81:         invdyn.train()
    82: 
    83:         n_gradient_step = 0
    84:         log = {"avg_loss_diffusion": 0.,  "avg_loss_invdyn": 0.}
    85: 
    86:         for batch in loop_dataloader(dataloader):
    87: 
    88:             obs = batch["obs"]["state"].to(args.device)
    89:             act = batch["act"].to(args.device)
    90:             val = batch["val"].to(args.device) / return_scale
    91: 
    92:             # ----------- Gradient Step ------------
    93:             log["avg_loss_diffusion"] += agent.update(obs, val)['loss']
    94:             diffusion_lr_scheduler.step()
    95:             if n_gradient_step <= args.invdyn_gradient_steps:
    96:                 log["avg_loss_invdyn"] += invdyn.update(obs[:, :-1], act[:, :-1], obs[:, 1:])['loss']
    97:                 invdyn_lr_scheduler.step()
    98: 
    99:             # ----------- Logging ------------
   100:             if (n_gradient_step + 1) % args.log_interval == 0:
   101:                 log["gradient_steps"] = n_gradient_step + 1
   102:                 log["avg_loss_diffusion"] /= args.log_interval
   103:                 log["avg_loss_invdyn"] /= args.log_interval
   104:                 print(log)
   105:                 log = {"avg_loss_diffusion": 0., "avg_loss_invdyn": 0.}
   106: 
   107:             # ----------- Saving ------------
   108:             if (n_gradient_step + 1) % args.save_interval == 0:
   109:                 agent.save(save_path + f"diffusion_ckpt_{n_gradient_step + 1}.pt")
   110:                 invdyn.save(save_path + f"invdyn_ckpt_{n_gradient_step + 1}.pt")
   111:                 agent.save(save_path + f"diffusion_ckpt_latest.pt")
   112:                 invdyn.save(save_path + f"invdyn_ckpt_latest.pt")
   113: 
   114:             n_gradient_step += 1
   115:             if n_gradient_step >= args.diffusion_gradient_steps:
   116:                 break
   117: 
   118:     # ---------------------- Finetune (placeholder for adaptdiffuser) ----------------------
   119:     elif args.mode == "finetune":
   120:         pass
   121: 
   122:     # ---------------------- Inference ----------------------
   123:     elif args.mode == "inference":
   124: 
   125:         # ============================================================================
   126:         # EDITABLE REGION 5: Inference Setup (lines 186-197)
   127:         # ============================================================================
   128: 
   129:         agent.load(save_path + f"diffusion_ckpt_{args.diffusion_ckpt}.pt")
   130:         agent.eval()
   131:         invdyn.load(save_path + f"invdyn_ckpt_{args.invdyn_ckpt}.pt")
   132:         invdyn.eval()
   133: 
   134:         # ============================================================================
   135:         # FIXED: Environment Setup (lines 198-206)
   136:         # ============================================================================

Lines 142–149:
   139:         normalizer = dataset.get_normalizer()
   140:         episode_rewards = []
   141: 
   142:         # ============================================================================
   143:         # EDITABLE REGION 6: Prior + Condition Initialization (lines 207-222)
   144:         # ============================================================================
   145: 
   146:         prior = torch.zeros((args.num_envs, args.task.horizon, obs_dim), device=args.device)
   147:         condition = torch.ones((args.num_envs, 1), device=args.device) * args.task.target_return
   148: 
   149:         for i in range(args.num_episodes):
   150:             env_eval.seed(args.seed + i * args.num_envs) if hasattr(env_eval, "seed") else None; obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0
   151: 
   152:             while not np.all(cum_done) and t < 1000 + 1:

Lines 161–180:
   158:                 # normalize obs
   159:                 obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
   160: 
   161:                 # ============================================================================
   162:                 # EDITABLE REGION 7: Action Sampling (lines 226-240)
   163:                 # ============================================================================
   164: 
   165:                 # sample trajectories
   166:                 prior[:, 0] = obs
   167:                 traj, log = agent.sample(
   168:                     prior, solver=args.solver,
   169:                     n_samples=args.num_envs, sample_steps=args.sampling_steps, use_ema=args.use_ema,
   170:                     condition_cfg=condition, w_cfg=args.task.w_cfg, temperature=args.temperature)
   171: 
   172:                 # inverse dynamic
   173:                 with torch.no_grad():
   174:                     act = invdyn.predict(obs, traj[:, 1, :]).cpu().numpy()
   175: 
   176:                 # FIXED post-sample print (outside editable range) references logp/idx;
   177:                 # DD has no candidate re-ranking, so define harmless placeholders.
   178:                 logp = torch.zeros((1, args.num_envs), device=args.device)
   179:                 idx = torch.zeros((args.num_envs,), dtype=torch.long, device=args.device)
   180: 
   181:                 # ============================================================================
   182:                 # FIXED: Environment Step + Reward Collection (lines 241-252)
   183:                 # ============================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
