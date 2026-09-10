"""Classifier-Free Guidance (CFG) baseline — minimal ablation of the default.

Reference: "Classifier-Free Diffusion Guidance" (Ho & Salimans, 2022),
https://arxiv.org/abs/2207.12598.

This baseline keeps EVERY architectural choice of the default classifier-guided
template and ONLY swaps the guidance pathway:
  - Same JannerUNet1d backbone, same DiscreteDiffusionSDE, same obs+act
    trajectory diffusion, same diffusion_steps and horizons.
  - Removes the CumRewClassifier (no classifier gradient term, w_cg=0).
  - Adds an MLPCondition that ingests a normalized return and is trained with
    label dropout (Ho & Salimans), then enables CFG via w_cfg at sample time.
  - Sampling uses w_cfg from the task config (target_return-conditioned). No
    candidate re-ranking — actions are read directly from the conditional
    sample, mirroring the standard CFG inference protocol.

This is the clean apples-to-apples comparison to the CG default. The Decision
Diffuser baseline (decision_diffuser.edit.py) is a separate, stronger CFG point
that also changes architecture, so the two together let the leaderboard
disentangle "CG vs CFG with same backbone" from "DD architecture upgrade".

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "CleanDiffuser/pipelines/custom_guidance.py"

_IMPORTS = """\
import os

import d4rl
import gym
import hydra
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_condition import MLPCondition
from cleandiffuser.nn_diffusion import JannerUNet1d
from cleandiffuser.utils import report_parameters, DD_RETURN_SCALE
from utils import set_seed
"""

_NETWORK_AGENT = """\
    # ============================================================================
    # cfg: Network + Agent Setup (JannerUNet1d + MLPCondition, no classifier)
    # ============================================================================

    return_scale = DD_RETURN_SCALE[args.task.env_name]

    # --------------- Network Architecture (same as default) -----------------
    nn_diffusion = JannerUNet1d(
        obs_dim + act_dim, model_dim=args.model_dim, emb_dim=args.model_dim, dim_mult=args.task.dim_mult,
        timestep_emb_type="positional", attention=False, kernel_size=5)

    # --------------- Classifier-Free condition network (replaces classifier) -----------------
    nn_condition = MLPCondition(
        in_dim=1, out_dim=args.model_dim,
        hidden_dims=[args.model_dim, ], act=nn.SiLU(), dropout=args.label_dropout)

    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    # report_parameters(nn_condition) crashes when nn_condition has fewer params than its
    # hardcoded top-K (sorted_keys[i] out of range for the tiny MLPCondition). Skip and
    # just print the total so logs stay informative.
    print(f"======================= Condition Network: MLPCondition =======================")
    print(f"Total parameters: {sum(p.numel() for p in nn_condition.parameters())}")
    print(f"==============================================================================")

    # ----------------- Masking (identical to default) -------------------
    fix_mask = torch.zeros((args.task.horizon, obs_dim + act_dim))
    fix_mask[0, :obs_dim] = 1.
    loss_weight = torch.ones((args.task.horizon, obs_dim + act_dim))
    loss_weight[0, obs_dim:] = args.action_loss_weight

    # --------------- Diffusion Model (same SDE, classifier=None, condition=MLP) --------------------
    agent = DiscreteDiffusionSDE(
        nn_diffusion, nn_condition,
        fix_mask=fix_mask, loss_weight=loss_weight, classifier=None, ema_rate=args.ema_rate,
        device=args.device, diffusion_steps=args.diffusion_steps, predict_noise=args.predict_noise)
"""

_TRAINING = """\
    # ============================================================================
    # cfg: Training (diffusion only with return-conditioning, no classifier)
    # ============================================================================

    # ---------------------- Training ----------------------
    if args.mode == "train":

        diffusion_lr_scheduler = CosineAnnealingLR(agent.optimizer, args.diffusion_gradient_steps)

        agent.train()

        n_gradient_step = 0
        log = {"avg_loss_diffusion": 0.}

        for batch in loop_dataloader(dataloader):

            obs = batch["obs"]["state"].to(args.device)
            act = batch["act"].to(args.device)
            val = batch["val"].to(args.device) / return_scale

            x = torch.cat([obs, act], -1)

            # ----------- Gradient Step (CFG-style: condition with label dropout) ------------
            log["avg_loss_diffusion"] += agent.update(x, val)['loss']
            diffusion_lr_scheduler.step()

            # ----------- Logging ------------
            if (n_gradient_step + 1) % args.log_interval == 0:
                log["gradient_steps"] = n_gradient_step + 1
                log["avg_loss_diffusion"] /= args.log_interval
                print(log)
                log = {"avg_loss_diffusion": 0.}

            # ----------- Saving ------------
            if (n_gradient_step + 1) % args.save_interval == 0:
                agent.save(save_path + f"diffusion_ckpt_{n_gradient_step + 1}.pt")
                agent.save(save_path + f"diffusion_ckpt_latest.pt")

            n_gradient_step += 1
            if n_gradient_step >= args.diffusion_gradient_steps:
                break

    # ---------------------- Finetune (placeholder) ----------------------
    elif args.mode == "finetune":
        pass
"""

_INFERENCE_SETUP = """\
        # ============================================================================
        # cfg: Inference Setup (diffusion only)
        # ============================================================================

        agent.load(save_path + f"diffusion_ckpt_{args.ckpt}.pt")

        agent.eval()
"""

_PRIOR_CONDITION = """\
        # ============================================================================
        # cfg: Prior + Condition Initialization (return-conditioned via MLPCondition)
        # ============================================================================

        prior = torch.zeros((args.num_envs, args.task.horizon, obs_dim + act_dim), device=args.device)
        condition = torch.ones((args.num_envs, 1), device=args.device) * args.task.target_return

        for i in range(args.num_episodes):
"""

_ACTION_SAMPLING = """\
                # ============================================================================
                # cfg: Action Sampling (CFG, no candidate re-ranking)
                # ============================================================================

                # sample trajectories conditioned on target return
                prior[:, 0, :obs_dim] = obs
                traj, log = agent.sample(
                    prior,
                    solver=args.solver,
                    n_samples=args.num_envs,
                    sample_steps=args.sampling_steps,
                    use_ema=args.use_ema,
                    condition_cfg=condition,
                    w_cfg=args.task.w_cfg,
                    w_cg=0.0,
                    temperature=args.temperature)

                # read actions directly from the conditional sample (no logp re-ranking)
                act = traj[:, 0, obs_dim:]
                act = act.clip(-1., 1.).cpu().numpy()

                # FIXED post-sample print (outside editable range) references logp/idx;
                # CFG has no candidate re-ranking, so define harmless placeholders.
                logp = torch.zeros((1, args.num_envs), device=args.device)
                idx = torch.zeros((args.num_envs,), dtype=torch.long, device=args.device)
"""

# Ordered bottom-to-top so line numbers remain stable across ops.
OPS = [
    # 7. Replace action sampling (lines 164-182)
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 164,
        "end_line": 182,
        "content": _ACTION_SAMPLING,
    },
    # 6. Replace prior + condition (lines 145-155)
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 145,
        "end_line": 152,
        "content": _PRIOR_CONDITION,
    },
    # 5. Replace inference setup (lines 128-135)
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 128,
        "end_line": 135,
        "content": _INFERENCE_SETUP,
    },
    # 4. Replace training + finetune (lines 72-123)
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 72,
        "end_line": 123,
        "content": _TRAINING,
    },
    # 3. Replace network + agent (lines 38-71)
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 38,
        "end_line": 71,
        "content": _NETWORK_AGENT,
    },
    # 1. Replace imports (lines 1-18)
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 1,
        "end_line": 18,
        "content": _IMPORTS,
    },
]
