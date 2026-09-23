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
- The **sampling algorithm itself** — the body of `sample_actions(policy, prior, obs, args)` (the `EDITABLE REGION: Sampling Algorithm` block) in `CleanDiffuser/pipelines/custom_sampling_method.py`, which turns a prior and an observation into actions. The default delegates to CleanDiffuser's built-in solvers (`policy.sample`); you may instead write the reverse process yourself, calling the denoiser (`policy.denoiser(x_t, t, cond)`, with `cond = policy.condition(obs)` and the schedule in `policy.alpha` / `policy.sigma`) directly with whatever discretization, step schedule, order or correction you want.
- `solver` and `sampling_steps` in `CleanDiffuser/configs/custom/mujoco/mujoco.yaml`, which drive the default implementation.

## What Is Fixed
- The actor and critic architectures, the training objective and the training loop
- `diffusion_steps`, training budgets, checkpoint selection, and EMA use
- Candidate selection, the environments, seeds, and vectorized evaluation loop

NFE is **measured, not declared**: fixed code outside `sample_actions` wraps the denoiser in a counter and hands the sampler only that counted `policy` facade (never the actor, the critic or the raw network), and reports the average number of network evaluations per action sample. `policy.denoiser`, directly or through `policy.sample`, is the only diffusion network the sampler may evaluate, and a sampler that makes no evaluation is rejected. Writing your own reverse process is therefore in scope — a sampler that spends 40 evaluations is scored as 40 no matter what any config field says, and one that reaches the same return in 10 is scored as 10.

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
- editable lines **52–74**
- `CleanDiffuser/configs/custom/mujoco/mujoco.yaml`
- editable lines **15–15**
- editable lines **17–17**


Other files you may **read** for context (do not modify):
- `CleanDiffuser/pipelines/custom_sampling_method.py`


## Readable Context


### `CleanDiffuser/pipelines/custom_sampling_method.py`  [EDITABLE — lines 52–74 only]

```python
     1: import os
     2: from copy import deepcopy
     3: from types import SimpleNamespace
     4: 
     5: import d4rl
     6: import gym
     7: import hydra
     8: import numpy as np
     9: import torch
    10: import torch.nn.functional as F
    11: from omegaconf import OmegaConf
    12: from torch.optim.lr_scheduler import CosineAnnealingLR
    13: from torch.utils.data import DataLoader
    14: 
    15: from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoTDDataset
    16: from cleandiffuser.dataset.dataset_utils import loop_dataloader
    17: from cleandiffuser.diffusion import DiscreteDiffusionSDE
    18: from cleandiffuser.nn_condition import IdentityCondition
    19: from cleandiffuser.nn_diffusion import DQLMlp
    20: from cleandiffuser.utils import report_parameters, DQLCritic, FreezeModules
    21: from utils import set_seed
    22: 
    23: 
    24: # ============================================================================
    25: # The sampler: `sample_actions` is called once per environment step
    26: # ============================================================================
    27: # It receives only what a sampler needs, never the actor, the critic or the
    28: # raw denoiser:
    29: #
    30: #   policy.denoiser(x_t, t, cond)  one network evaluation: eps or x0 prediction
    31: #                                  (per policy.predict_noise) for integer
    32: #                                  timesteps t of shape [batch]; this is the
    33: #                                  NFE the score counts
    34: #   policy.condition(obs)          the (fixed) condition encoder: obs -> cond
    35: #   policy.sample(prior, ...)      CleanDiffuser's built-in solvers
    36: #                                  (DiscreteDiffusionSDE.sample arguments);
    37: #                                  each denoiser call inside is counted too
    38: #   policy.alpha, policy.sigma, policy.logSNR, policy.t_diffusion
    39: #                                  the noise schedule over the
    40: #                                  policy.diffusion_steps training timesteps
    41: #   policy.x_min, policy.x_max, policy.clip_prediction(pred, x_t, alpha, sigma)
    42: #
    43: # `prior` is the zero prior [num_envs * num_candidates, act_dim], `obs` the
    44: # normalized observations repeated per candidate, `args` a read-only copy of
    45: # the config. Return `act` of shape [num_envs * num_candidates, act_dim].
    46: # The only diffusion network you may evaluate is `policy.denoiser` (directly
    47: # or through `policy.sample`); a sampler that makes no evaluation is rejected.
    48: def sample_actions(policy, prior, obs, args):
    49:     # ========================================================================
    50:     # EDITABLE REGION: Sampling Algorithm
    51:     # ========================================================================
    52:     # Produce `act` of shape [num_envs * num_candidates, act_dim] by running a
    53:     # reverse diffusion process conditioned on `obs`.
    54:     #
    55:     # The default below delegates to CleanDiffuser's built-in solvers, driven
    56:     # by `solver` / `sampling_steps` in the YAML. You are not limited to that:
    57:     # implement the reverse process yourself and call the denoiser directly —
    58:     #
    59:     #   cond = policy.condition(obs)
    60:     #   pred = policy.denoiser(x_t, t, cond)   # eps or x0, per policy.predict_noise
    61:     #
    62:     # with the schedule in policy.alpha / policy.sigma (see
    63:     # cleandiffuser/diffusion/diffusionsde.py). Fewer evaluations at the same
    64:     # return is the point: every call to the denoiser is counted and reported
    65:     # as the NFE the score penalizes, so the cost you pay is the cost you are
    66:     # scored on.
    67:     act, log = policy.sample(
    68:         prior,
    69:         solver=args.solver,
    70:         n_samples=args.num_envs * args.num_candidates,
    71:         sample_steps=args.sampling_steps,
    72:         condition_cfg=obs, w_cfg=1.0,
    73:         use_ema=args.use_ema, temperature=args.temperature)
    74:     return act
    75:     # ========================================================================
    76:     # END EDITABLE REGION
    77:     # ========================================================================
    78: 
    79: 
    80: @hydra.main(config_path="../configs/custom/mujoco", config_name="mujoco", version_base=None)
    81: def pipeline(args):
    82: 
    83:     set_seed(args.seed)
    84: 
    85:     save_path = f'results/{args.pipeline_name}/{args.task.env_name}_s{args.seed}/'  # per-seed dir: parallel seeds of one label must not share ckpts
    86:     if args.mode == "train": import shutil; shutil.rmtree(save_path, ignore_errors=True)  # fresh train dir: never silently score a stale exact-step ckpt
    87:     os.makedirs(save_path, exist_ok=True)
    88: 
    89:     # ---------------------- Create Dataset ----------------------
    90:     env = gym.make(args.task.env_name)
    91:     dataset = D4RLMuJoCoTDDataset(d4rl.qlearning_dataset(env), args.normalize_reward)
    92:     dataloader = DataLoader(
    93:         dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    94:     obs_dim, act_dim = dataset.o_dim, dataset.a_dim
    95: 
    96:     # ============================================================================
    97:     # FIXED: Policy, Critic and Training
    98:     # ============================================================================
    99:     # Diffusion Q-Learning (DQL): diffusion actor + twin Q critic with BC + Q
   100:     # loss. The trained actor/critic, dataset, environment list, seeds and
   101:     # evaluation loop are fixed — this task is about the sampler, so the thing
   102:     # you edit is the reverse process at inference time: `sample_actions` above.
   103: 
   104:     # --------------- Network Architecture -----------------
   105:     nn_diffusion = DQLMlp(obs_dim, act_dim, emb_dim=64, timestep_emb_type="positional").to(args.device)
   106:     nn_condition = IdentityCondition(dropout=0.0).to(args.device)
   107: 
   108:     print(f"======================= Parameter Report of Diffusion Model =======================")
   109:     report_parameters(nn_diffusion)
   110:     print(f"==============================================================================")
   111: 
   112:     # --------------- Diffusion Model Actor --------------------
   113:     actor = DiscreteDiffusionSDE(
   114:         nn_diffusion, nn_condition, predict_noise=args.predict_noise, optim_params={"lr": args.actor_learning_rate},
   115:         x_max=+1. * torch.ones((1, act_dim), device=args.device),
   116:         x_min=-1. * torch.ones((1, act_dim), device=args.device),
   117:         diffusion_steps=args.diffusion_steps, ema_rate=args.ema_rate, device=args.device)
   118: 
   119:     # ------------------ Critic ---------------------
   120:     critic = DQLCritic(obs_dim, act_dim, hidden_dim=args.hidden_dim).to(args.device)
   121:     critic_target = deepcopy(critic).requires_grad_(False).eval()
   122:     critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_learning_rate)
   123: 
   124:     # ---------------------- Training ----------------------
   125:     if args.mode == "train":
   126: 
   127:         actor_lr_scheduler = CosineAnnealingLR(actor.optimizer, T_max=args.gradient_steps)
   128:         critic_lr_scheduler = CosineAnnealingLR(critic_optim, T_max=args.gradient_steps)
   129: 
   130:         actor.train()
   131:         critic.train()
   132: 
   133:         n_gradient_step = 0
   134:         log = {"bc_loss": 0., "q_loss": 0., "critic_loss": 0., "target_q_mean": 0.}
   135: 
   136:         prior = torch.zeros((args.batch_size, act_dim), device=args.device)
   137: 
   138:         for batch in loop_dataloader(dataloader):
   139: 
   140:             obs, next_obs = batch["obs"]["state"].to(args.device), batch["next_obs"]["state"].to(args.device)
   141:             act = batch["act"].to(args.device)
   142:             rew = batch["rew"].to(args.device)
   143:             tml = batch["tml"].to(args.device)
   144: 
   145:             # Critic Training
   146:             current_q1, current_q2 = critic(obs, act)
   147: 
   148:             next_act, _ = actor.sample(
   149:                 prior, solver=args.solver,
   150:                 n_samples=args.batch_size, sample_steps=args.sampling_steps, use_ema=True,
   151:                 temperature=1.0, condition_cfg=next_obs, w_cfg=1.0, requires_grad=False)
   152: 
   153:             target_q = torch.min(*critic_target(next_obs, next_act))
   154:             target_q = (rew + (1 - tml) * args.discount * target_q).detach()
   155: 
   156:             critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
   157: 
   158:             critic_optim.zero_grad()
   159:             critic_loss.backward()
   160:             critic_optim.step()
   161: 
   162:             # Policy Training
   163:             bc_loss = actor.loss(act, obs)
   164:             new_act, _ = actor.sample(
   165:                 prior, solver=args.solver,
   166:                 n_samples=args.batch_size, sample_steps=args.sampling_steps, use_ema=False,
   167:                 temperature=1.0, condition_cfg=obs, w_cfg=1.0, requires_grad=True)
   168: 
   169:             with FreezeModules([critic, ]):
   170:                 q1_new_action, q2_new_action = critic(obs, new_act)
   171:             if np.random.uniform() > 0.5:
   172:                 q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
   173:             else:
   174:                 q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
   175:             actor_loss = bc_loss + args.task.eta * q_loss
   176: 
   177:             actor.optimizer.zero_grad()
   178:             actor_loss.backward()
   179:             actor.optimizer.step()
   180: 
   181:             actor_lr_scheduler.step()
   182:             critic_lr_scheduler.step()
   183: 
   184:             # ema
   185:             if n_gradient_step % args.ema_update_interval == 0:
   186:                 if n_gradient_step >= 1000:
   187:                     actor.ema_update()
   188:                 for param, target_param in zip(critic.parameters(), critic_target.parameters()):
   189:                     target_param.data.copy_(0.995 * param.data + (1 - 0.995) * target_param.data)
   190: 
   191:             log["bc_loss"] += bc_loss.item()
   192:             log["q_loss"] += q_loss.item()
   193:             log["critic_loss"] += critic_loss.item()
   194:             log["target_q_mean"] += target_q.mean().item()
   195: 
   196:             if (n_gradient_step + 1) % args.log_interval == 0:
   197:                 log["gradient_steps"] = n_gradient_step + 1
   198:                 log["bc_loss"] /= args.log_interval
   199:                 log["q_loss"] /= args.log_interval
   200:                 log["critic_loss"] /= args.log_interval
   201:                 log["target_q_mean"] /= args.log_interval
   202:                 print(f"TRAIN_METRICS gradient_steps={log['gradient_steps']} "
   203:                       f"bc_loss={log['bc_loss']:.4f} q_loss={log['q_loss']:.4f} "
   204:                       f"critic_loss={log['critic_loss']:.4f} target_q_mean={log['target_q_mean']:.4f}")
   205:                 log = {"bc_loss": 0., "q_loss": 0., "critic_loss": 0., "target_q_mean": 0.}
   206: 
   207:             if (n_gradient_step + 1) % args.save_interval == 0:
   208:                 actor.save(save_path + f"diffusion_ckpt_{n_gradient_step + 1}.pt")
   209:                 actor.save(save_path + f"diffusion_ckpt_latest.pt")
   210:                 torch.save({
   211:                     "critic": critic.state_dict(),
   212:                     "critic_target": critic_target.state_dict(),
   213:                 }, save_path + f"critic_ckpt_{n_gradient_step + 1}.pt")
   214:                 torch.save({
   215:                     "critic": critic.state_dict(),
   216:                     "critic_target": critic_target.state_dict(),
   217:                 }, save_path + f"critic_ckpt_latest.pt")
   218: 
   219:             n_gradient_step += 1
   220:             if n_gradient_step >= args.gradient_steps:
   221:                 break
   222: 
   223:     # ---------------------- Inference ----------------------
   224:     elif args.mode == "inference":
   225: 
   226:         actor.load(save_path + f"diffusion_ckpt_{args.ckpt}.pt")
   227:         critic_ckpt = torch.load(save_path + f"critic_ckpt_{args.ckpt}.pt")
   228:         critic.load_state_dict(critic_ckpt["critic"])
   229:         critic_target.load_state_dict(critic_ckpt["critic_target"])
   230: 
   231:         actor.eval()
   232:         critic.eval()
   233:         critic_target.eval()
   234: 
   235:         env_eval = gym.vector.make(args.task.env_name, args.num_envs)
   236:         normalizer = dataset.get_normalizer()
   237:         episode_rewards = []
   238: 
   239:         # ============================================================================
   240:         # FIXED: NFE accounting — do not modify
   241:         # ============================================================================
   242:         # Counts real denoiser evaluations, so the reported NFE is what your
   243:         # sampler actually spends rather than a number declared in a config
   244:         # file. Both denoisers are swapped for counting wrappers that hold the
   245:         # network only in a closure, and `sample_actions` receives a facade
   246:         # built from them — never the actor, the critic or the raw network — so
   247:         # every evaluation it can make goes through the counter, which lives
   248:         # here and nowhere the sampler can reach. One call == one network
   249:         # evaluation, whatever batch it carries.
   250:         _nfe_calls = [0]
   251: 
   252:         def _counted(net):
   253:             class _CountedDenoiser(torch.nn.Module):
   254:                 def forward(self, x, t, condition=None):
   255:                     _nfe_calls[0] += 1
   256:                     return net(x, t, condition)
   257:             return _CountedDenoiser()
   258: 
   259:         actor.model["diffusion"] = _counted(actor.model["diffusion"])
   260:         actor.model_ema["diffusion"] = _counted(actor.model_ema["diffusion"])
   261:         _model = actor.model_ema if args.use_ema else actor.model
   262: 
   263:         def _policy_denoiser(x_t, t, condition=None):
   264:             return _model["diffusion"](x_t, t, condition)
   265: 
   266:         def _policy_condition(obs, mask=None):
   267:             return _model["condition"](obs, mask)
   268: 
   269:         def _policy_sample(prior, **kwargs):
   270:             return actor.sample(prior, **kwargs)
   271: 
   272:         def _policy_clip_prediction(pred, xt, alpha, sigma):
   273:             return actor.clip_prediction(pred, xt, alpha, sigma)
   274: 
   275:         def _const(v):
   276:             return v.detach().clone() if isinstance(v, torch.Tensor) else v
   277: 
   278:         policy = SimpleNamespace(
   279:             denoiser=_policy_denoiser, condition=_policy_condition, sample=_policy_sample,
   280:             clip_prediction=_policy_clip_prediction,
   281:             alpha=_const(actor.alpha), sigma=_const(actor.sigma), logSNR=_const(actor.logSNR),
   282:             t_diffusion=_const(actor.t_diffusion), diffusion_steps=actor.diffusion_steps,
   283:             predict_noise=actor.predict_noise, x_min=_const(actor.x_min), x_max=_const(actor.x_max),
   284:             fix_mask=_const(actor.fix_mask), device=actor.device)
   285:         sampler_args = deepcopy(args)
   286:         OmegaConf.set_readonly(sampler_args, True)
   287:         n_act_samples = 0
   288: 
   289:         prior = torch.zeros((args.num_envs * args.num_candidates, act_dim), device=args.device)
   290:         for i in range(args.num_episodes):
   291: 
   292:             env_eval.seed(args.seed + i * args.num_envs) if hasattr(env_eval, "seed") else None; obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0
   293: 
   294:             while not np.all(cum_done) and t < 1000 + 1:
   295:                 obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
   296:                 obs = obs.unsqueeze(1).repeat(1, args.num_candidates, 1).view(-1, obs_dim)
   297: 
   298:                 _calls_before = _nfe_calls[0]
   299:                 act = sample_actions(policy, prior.clone(), obs.clone(), sampler_args)
   300:                 if _nfe_calls[0] == _calls_before:
   301:                     raise RuntimeError("sample_actions made no denoiser evaluation: actions must come "
   302:                                        "from the diffusion policy through policy.denoiser / policy.sample")
   303:                 if not isinstance(act, torch.Tensor) or tuple(act.shape) != (args.num_envs * args.num_candidates, act_dim):
   304:                     raise RuntimeError(f"sample_actions must return a tensor of shape "
   305:                                        f"{(args.num_envs * args.num_candidates, act_dim)}, got "
   306:                                        f"{tuple(act.shape) if isinstance(act, torch.Tensor) else type(act)}")
   307:                 n_act_samples += 1
   308: 
   309:                 # ====================================================================
   310:                 # FIXED: Candidate Selection and Environment Step
   311:                 # ====================================================================
   312: 
   313:                 with torch.no_grad():
   314:                     q = critic_target.q_min(obs, act)
   315:                     q = q.view(-1, args.num_candidates, 1)
   316:                     w = torch.softmax(q * args.task.weight_temperature, 1)
   317:                     act = act.view(-1, args.num_candidates, act_dim)
   318: 
   319:                     indices = torch.multinomial(w.squeeze(-1), 1).squeeze(-1)
   320:                     sampled_act = act[torch.arange(act.shape[0]), indices].cpu().numpy()
   321: 
   322:                 obs, rew, done, info = env_eval.step(sampled_act)
   323: 
   324:                 t += 1
   325:                 cum_done = done if cum_done is None else np.logical_or(cum_done, done)
   326:                 ep_reward += (rew * (1 - cum_done)) if t < 1000 else rew
   327: 
   328:                 if np.all(cum_done):
   329:                     break
   330: 
   331:             episode_rewards.append(ep_reward)
   332: 
   333:         raw_episode_rewards = episode_rewards
   334:         episode_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
   335:         episode_rewards = np.array(episode_rewards)
   336:         mean_score = float(np.mean(episode_rewards))
   337:         std_score = float(np.std(episode_rewards))
   338:         mean_ep_reward = float(np.mean(raw_episode_rewards))
   339:         print(f"EVAL_METRICS normalized_score={mean_score:.4f} normalized_score_std={std_score:.4f} episode_reward={mean_ep_reward:.2f}")
   340: 
   341:         # Ceil, not round: an adaptive sampler averaging 10.5 evaluations must
   342:         # not report 10 and collect the full no-penalty credit reserved for a
   343:         # 10-step budget. Round-half-to-even would also floor an average of 0.5
   344:         # to zero. Constant-call samplers are unaffected.
   345:         measured_nfe = -(-_nfe_calls[0] // max(n_act_samples, 1))
   346:         print(f"NFE_METRICS sampling_steps={measured_nfe}", flush=True)
   347: 
   348:     else:
   349:         raise ValueError(f"Invalid mode: {args.mode}")
   350: 
   351: 
   352: if __name__ == "__main__":
   353:     pipeline()
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
