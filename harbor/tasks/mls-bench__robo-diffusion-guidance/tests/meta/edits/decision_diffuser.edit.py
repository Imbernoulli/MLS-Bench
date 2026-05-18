"""Decision Diffuser baseline — full DD port (alternative architecture).

Reference: "Is Conditional Generative Modeling all you need for Decision-Making?"
(Ajay et al., 2022), https://arxiv.org/abs/2211.15657.

This is NOT a minimal CFG ablation of the default; it is a verbatim port of
cleandiffuser's dd_d4rl_mujoco.py and changes the entire pipeline:
  - State-only diffusion (obs, not obs+act)
  - DiT1d Transformer replaces JannerUNet1d
  - Return conditioning via MLPCondition (no CumRewClassifier)
  - MlpInvDynamic to recover actions from state transitions
  - ContinuousDiffusionSDE instead of DiscreteDiffusionSDE

Provided as a strong CFG-based reference alongside the JannerUNet1d-backbone
baselines (default=CG, cfg=minimal CFG ablation, no_guidance=unconditional).

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
from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.invdynamic import MlpInvDynamic
from cleandiffuser.nn_condition import MLPCondition
from cleandiffuser.nn_diffusion import DiT1d
from cleandiffuser.utils import report_parameters, DD_RETURN_SCALE
from utils import set_seed
"""

_PRE_DATASET = """\
@hydra.main(config_path="../configs/dd/mujoco", config_name="mujoco", version_base=None)
def pipeline(args):

    return_scale = DD_RETURN_SCALE[args.task.env_name]

    set_seed(args.seed)

    save_path = f'results/{args.pipeline_name}/{args.task.env_name}/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
"""

_NETWORK_AGENT = """\
    # ============================================================================
    # EDITABLE REGION 3: Network + Agent Setup (lines 40-72)
    # ============================================================================

    # --------------- Network Architecture -----------------
    nn_diffusion = DiT1d(
        obs_dim, emb_dim=args.emb_dim,
        d_model=args.d_model, n_heads=args.n_heads, depth=args.depth, timestep_emb_type="fourier")
    nn_condition = MLPCondition(
        in_dim=1, out_dim=args.emb_dim, hidden_dims=[args.emb_dim, ], act=nn.SiLU(), dropout=args.label_dropout)

    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"==============================================================================")

    # ----------------- Masking -------------------
    fix_mask = torch.zeros((args.task.horizon, obs_dim))
    fix_mask[0] = 1.
    loss_weight = torch.ones((args.task.horizon, obs_dim))
    loss_weight[1] = args.next_obs_loss_weight

    # --------------- Diffusion Model with Classifier-Free Guidance --------------------
    agent = ContinuousDiffusionSDE(
        nn_diffusion, nn_condition,
        fix_mask=fix_mask, loss_weight=loss_weight, ema_rate=args.ema_rate,
        device=args.device, predict_noise=args.predict_noise, noise_schedule="linear")

    # --------------- Inverse Dynamic -------------------
    invdyn = MlpInvDynamic(obs_dim, act_dim, 512, nn.Tanh(), {"lr": 2e-4}, device=args.device)
"""

_TRAINING = """\
    # ============================================================================
    # EDITABLE REGION 4: Training + Finetune (lines 74-182)
    # ============================================================================

    # ---------------------- Training ----------------------
    if args.mode == "train":

        diffusion_lr_scheduler = CosineAnnealingLR(agent.optimizer, args.diffusion_gradient_steps)
        invdyn_lr_scheduler = CosineAnnealingLR(invdyn.optim, args.invdyn_gradient_steps)

        agent.train()
        invdyn.train()

        n_gradient_step = 0
        log = {"avg_loss_diffusion": 0.,  "avg_loss_invdyn": 0.}

        for batch in loop_dataloader(dataloader):

            obs = batch["obs"]["state"].to(args.device)
            act = batch["act"].to(args.device)
            val = batch["val"].to(args.device) / return_scale

            # ----------- Gradient Step ------------
            log["avg_loss_diffusion"] += agent.update(obs, val)['loss']
            diffusion_lr_scheduler.step()
            if n_gradient_step <= args.invdyn_gradient_steps:
                log["avg_loss_invdyn"] += invdyn.update(obs[:, :-1], act[:, :-1], obs[:, 1:])['loss']
                invdyn_lr_scheduler.step()

            # ----------- Logging ------------
            if (n_gradient_step + 1) % args.log_interval == 0:
                log["gradient_steps"] = n_gradient_step + 1
                log["avg_loss_diffusion"] /= args.log_interval
                log["avg_loss_invdyn"] /= args.log_interval
                print(log)
                log = {"avg_loss_diffusion": 0., "avg_loss_invdyn": 0.}

            # ----------- Saving ------------
            if (n_gradient_step + 1) % args.save_interval == 0:
                agent.save(save_path + f"diffusion_ckpt_{n_gradient_step + 1}.pt")
                invdyn.save(save_path + f"invdyn_ckpt_{n_gradient_step + 1}.pt")
                agent.save(save_path + f"diffusion_ckpt_latest.pt")
                invdyn.save(save_path + f"invdyn_ckpt_latest.pt")

            n_gradient_step += 1
            if n_gradient_step >= args.diffusion_gradient_steps:
                break

    # ---------------------- Finetune (placeholder for adaptdiffuser) ----------------------
    elif args.mode == "finetune":
        pass
"""

_INFERENCE_SETUP = """\
        # ============================================================================
        # EDITABLE REGION 5: Inference Setup (lines 186-197)
        # ============================================================================

        agent.load(save_path + f"diffusion_ckpt_{args.diffusion_ckpt}.pt")
        agent.eval()
        invdyn.load(save_path + f"invdyn_ckpt_{args.invdyn_ckpt}.pt")
        invdyn.eval()
"""

_PRIOR_CONDITION = """\
        # ============================================================================
        # EDITABLE REGION 6: Prior + Condition Initialization (lines 207-222)
        # ============================================================================

        prior = torch.zeros((args.num_envs, args.task.horizon, obs_dim), device=args.device)
        condition = torch.ones((args.num_envs, 1), device=args.device) * args.task.target_return

        for i in range(args.num_episodes):
"""

_ACTION_SAMPLING = """\
                # ============================================================================
                # EDITABLE REGION 7: Action Sampling (lines 226-240)
                # ============================================================================

                # sample trajectories
                prior[:, 0] = obs
                traj, log = agent.sample(
                    prior, solver=args.solver,
                    n_samples=args.num_envs, sample_steps=args.sampling_steps, use_ema=args.use_ema,
                    condition_cfg=condition, w_cfg=args.task.w_cfg, temperature=args.temperature)

                # inverse dynamic
                with torch.no_grad():
                    act = invdyn.predict(obs, traj[:, 1, :]).cpu().numpy()

                # FIXED post-sample print (outside editable range) references logp/idx;
                # DD has no candidate re-ranking, so define harmless placeholders.
                logp = torch.zeros((1, args.num_envs), device=args.device)
                idx = torch.zeros((args.num_envs,), dtype=torch.long, device=args.device)
"""

# Ordered bottom-to-top so line numbers remain stable across ops.
OPS = [
    # 7. Replace action sampling (lines 164-182) — bottom first
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
    # 2. Replace config path + return_scale (lines 21-28)
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 21,
        "end_line": 28,
        "content": _PRE_DATASET,
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
