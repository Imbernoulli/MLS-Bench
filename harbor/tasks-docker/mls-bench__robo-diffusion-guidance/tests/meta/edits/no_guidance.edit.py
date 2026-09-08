"""No-Guidance baseline — unconditional trajectory diffusion.

An ablation of Diffuser that removes classifier guidance entirely:
  - No CumRewClassifier trained alongside the diffusion model.
  - No classifier gradient term mixed into the reverse process at sample time.
  - No log-probability-based trajectory re-ranking.

Architecturally the diffusion UNet and training objective match the CG template
(same JannerUNet1d on obs+act trajectories, same DiscreteDiffusionSDE). Only the
guidance pathway is stripped. This is the reference point a meaningful guidance
strategy must beat.

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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_diffusion import JannerUNet1d
from cleandiffuser.utils import report_parameters
from utils import set_seed
"""

_NETWORK_AGENT = """\
    # ============================================================================
    # no_guidance: Network + Agent Setup (no classifier)
    # ============================================================================

    nn_diffusion = JannerUNet1d(
        obs_dim + act_dim, model_dim=args.model_dim, emb_dim=args.model_dim, dim_mult=args.task.dim_mult,
        timestep_emb_type="positional", attention=False, kernel_size=5)

    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"==============================================================================")

    fix_mask = torch.zeros((args.task.horizon, obs_dim + act_dim))
    fix_mask[0, :obs_dim] = 1.
    loss_weight = torch.ones((args.task.horizon, obs_dim + act_dim))
    loss_weight[0, obs_dim:] = args.action_loss_weight

    agent = DiscreteDiffusionSDE(
        nn_diffusion, None,
        fix_mask=fix_mask, loss_weight=loss_weight, classifier=None, ema_rate=args.ema_rate,
        device=args.device, diffusion_steps=args.diffusion_steps, predict_noise=args.predict_noise)

"""

_TRAINING = """\
    # ============================================================================
    # no_guidance: Training (diffusion only, no classifier)
    # ============================================================================

    if args.mode == "train":

        diffusion_lr_scheduler = CosineAnnealingLR(agent.optimizer, args.diffusion_gradient_steps)

        agent.train()

        n_gradient_step = 0
        log = {"avg_loss_diffusion": 0.}

        for batch in loop_dataloader(dataloader):

            obs = batch["obs"]["state"].to(args.device)
            act = batch["act"].to(args.device)

            x = torch.cat([obs, act], -1)

            log["avg_loss_diffusion"] += agent.update(x)['loss']
            diffusion_lr_scheduler.step()

            if (n_gradient_step + 1) % args.log_interval == 0:
                log["gradient_steps"] = n_gradient_step + 1
                log["avg_loss_diffusion"] /= args.log_interval
                print(log)
                log = {"avg_loss_diffusion": 0.}

            if (n_gradient_step + 1) % args.save_interval == 0:
                agent.save(save_path + f"diffusion_ckpt_{n_gradient_step + 1}.pt")
                agent.save(save_path + f"diffusion_ckpt_latest.pt")

            n_gradient_step += 1
            if n_gradient_step >= args.diffusion_gradient_steps:
                break

    elif args.mode == "finetune":
        pass

"""

_INFERENCE_SETUP = """\
        # ============================================================================
        # no_guidance: Inference Setup (diffusion only)
        # ============================================================================

        agent.load(save_path + f"diffusion_ckpt_{args.ckpt}.pt")

        agent.eval()

"""

_PRIOR_CONDITION = """\
        # ============================================================================
        # no_guidance: Prior Initialization (no condition, no target return)
        # ============================================================================

        prior = torch.zeros((args.num_envs, args.task.horizon, obs_dim + act_dim), device=args.device)

        for i in range(args.num_episodes):
"""

_ACTION_SAMPLING = """\
                # ============================================================================
                # no_guidance: Action Sampling (w_cg=0, no re-ranking)
                # ============================================================================

                prior[:, 0, :obs_dim] = obs
                traj, log = agent.sample(
                    prior,
                    solver=args.solver,
                    n_samples=args.num_envs,
                    sample_steps=args.sampling_steps,
                    use_ema=args.use_ema,
                    w_cg=0.0,
                    w_cfg=0.0,
                    temperature=args.temperature)

                act = traj[:, 0, obs_dim:]
                act = act.clip(-1., 1.).cpu().numpy()

                # FIXED post-sample print (outside editable range) references logp/idx;
                # no_guidance has no candidate re-ranking, so define harmless placeholders.
                logp = torch.zeros((1, args.num_envs), device=args.device)
                idx = torch.zeros((args.num_envs,), dtype=torch.long, device=args.device)
"""

# Ordered bottom-to-top so line numbers remain stable.
OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 164,
        "end_line": 182,
        "content": _ACTION_SAMPLING,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 145,
        "end_line": 152,
        "content": _PRIOR_CONDITION,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 128,
        "end_line": 135,
        "content": _INFERENCE_SETUP,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 72,
        "end_line": 123,
        "content": _TRAINING,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 38,
        "end_line": 71,
        "content": _NETWORK_AGENT,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 1,
        "end_line": 18,
        "content": _IMPORTS,
    },
]
