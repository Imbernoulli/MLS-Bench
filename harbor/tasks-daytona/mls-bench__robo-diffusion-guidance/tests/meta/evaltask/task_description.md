# Robo-Diffusion: Guided Sampling Strategy Design

## Objective
Design one improved guidance mechanism for a fixed trajectory-level diffusion planner on offline D4RL MuJoCo benchmarks. This task is narrower than `robo-diffusion-policy`: the research question is how to condition or guide the reverse diffusion process, not how to redesign the whole planner or the model-free actor-critic training loop.

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
- Dataset and environment loop (D4RL MuJoCo, `env.get_dataset()`, `gym.vector.make`, normalization, reward collection).
- Environment names, seeds, and final D4RL score computation.
- Evaluation protocol: 10 envs × 10 episodes, `env.get_normalized_score`.
- Top-level training hyperparameters in `mujoco.yaml`: `diffusion_gradient_steps=100000`, `batch_size=256`, `model_dim=32`, `solver=ddpm`, `diffusion_steps=20`, `sampling_steps=20`.

## Evaluation
You will be evaluated on three D4RL (Fu et al., 2020, arXiv:2004.07219) MuJoCo environments:
1. **hopper-medium-v2**
2. **walker2d-medium-v2**
3. **halfcheetah-medium-v2**

Primary metric:
- **Normalized Score**: D4RL normalized score (higher is better), reported per env. The task aggregator is the geometric mean across the three envs.

Reported in the leaderboard alongside the per-env training wall-clock.

## Baselines
Four baselines are provided in `edits/`:

### 1. `default` — Diffuser (Classifier Guidance, CG)
- Unmodified template. `JannerUNet1d` + `CumRewClassifier`, `w_cg = {0.3, 0.007, 0.0001}` for hopper / walker2d / halfcheetah.
- Inference uses 64 candidates re-ranked by classifier log-prob.
- Reference: Janner et al., 2022, Diffuser, arXiv:2205.09991.

### 2. `cfg` — minimal CFG ablation of the default
- **Same** `JannerUNet1d` backbone, **same** `DiscreteDiffusionSDE`, **same** obs+act trajectory diffusion as `default`.
- Replaces `CumRewClassifier` with an `MLPCondition` over normalized return, trained with label dropout. Sampling uses `condition_cfg=target_return`, `w_cfg = {4.4, 6.0, 3.2}` (Decision Diffuser paper values), `w_cg = 0`. No candidate re-ranking.

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
