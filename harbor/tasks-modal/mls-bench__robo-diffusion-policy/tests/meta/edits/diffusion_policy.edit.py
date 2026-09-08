"""Diffusion Policy baseline: pure behavior cloning with a diffusion actor.

Reference: Diffusion Policy: Visuomotor Policy Learning via Action Diffusion
(Chi et al., 2023), https://arxiv.org/abs/2303.04137. Applied here to low-dim
D4RL MuJoCo, this reduces to the diffusion BC component of cleandiffuser's
dql_d4rl_mujoco.py with the twin-Q critic removed.

Concretely:
  - Same DQLMlp actor + DiscreteDiffusionSDE + IdentityCondition(obs) backbone
  - Training: BC loss only, no critic, no Q-target, no eta * q_loss term
  - Inference: sample one action per env, no candidate reranking
"""

_FILE = "CleanDiffuser/pipelines/custom_policy.py"

_ALGORITHM_TRAINING = """\
    # ============================================================================
    # diffusion_policy baseline: diffusion BC only, no critic / no reranking
    # ============================================================================

    # --------------- Network Architecture -----------------
    nn_diffusion = DQLMlp(obs_dim, act_dim, emb_dim=64, timestep_emb_type="positional").to(args.device)
    nn_condition = IdentityCondition(dropout=0.0).to(args.device)

    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"==============================================================================")

    # --------------- Diffusion Model Actor --------------------
    actor = DiscreteDiffusionSDE(
        nn_diffusion, nn_condition, predict_noise=args.predict_noise, optim_params={"lr": args.actor_learning_rate},
        x_max=+1. * torch.ones((1, act_dim), device=args.device),
        x_min=-1. * torch.ones((1, act_dim), device=args.device),
        diffusion_steps=args.diffusion_steps, ema_rate=args.ema_rate, device=args.device)

    # ---------------------- Training ----------------------
    if args.mode == "train":

        actor_lr_scheduler = CosineAnnealingLR(actor.optimizer, T_max=args.gradient_steps)

        actor.train()

        n_gradient_step = 0
        log = {"bc_loss": 0.}

        for batch in loop_dataloader(dataloader):

            obs = batch["obs"]["state"].to(args.device)
            act = batch["act"].to(args.device)

            bc_loss = actor.update(act, obs)["loss"]
            actor_lr_scheduler.step()

            if n_gradient_step % args.ema_update_interval == 0 and n_gradient_step >= 1000:
                actor.ema_update()

            log["bc_loss"] += bc_loss

            if (n_gradient_step + 1) % args.log_interval == 0:
                log["gradient_steps"] = n_gradient_step + 1
                log["bc_loss"] /= args.log_interval
                print(f"TRAIN_METRICS gradient_steps={log['gradient_steps']} bc_loss={log['bc_loss']:.4f}")
                log = {"bc_loss": 0.}

            if (n_gradient_step + 1) % args.save_interval == 0:
                actor.save(save_path + f"diffusion_ckpt_{n_gradient_step + 1}.pt")
                actor.save(save_path + f"diffusion_ckpt_latest.pt")

            n_gradient_step += 1
            if n_gradient_step >= args.gradient_steps:
                break
"""

_INFERENCE_SETUP = """\
        actor.load(save_path + f"diffusion_ckpt_{args.ckpt}.pt")
        actor.eval()
"""

_PRIOR = """\
        prior = torch.zeros((args.num_envs, act_dim), device=args.device)
"""

_ACTION_SELECTION = """\
                obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)

                act, _ = actor.sample(
                    prior,
                    solver=args.solver,
                    n_samples=args.num_envs,
                    sample_steps=args.sampling_steps,
                    condition_cfg=obs, w_cfg=1.0,
                    use_ema=args.use_ema, temperature=args.temperature)
                sampled_act = act.clip(-1., 1.).cpu().numpy()
"""

OPS = [
    # bottom-to-top: later original line numbers remain stable.
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 188,
        "end_line": 207,
        "content": _ACTION_SELECTION,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 182,
        "end_line": 182,
        "content": _PRIOR,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 169,
        "end_line": 176,
        "content": _INFERENCE_SETUP,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 38,
        "end_line": 165,
        "content": _ALGORITHM_TRAINING,
    },
]
