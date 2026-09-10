# Robo-Diffusion: Policy Algorithm Design

## Objective
Design a single model-free offline RL policy algorithm that uses a diffusion actor for action generation and improves D4RL MuJoCo control performance.

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

## Evaluation
Evaluated on three D4RL MuJoCo environments:
1. **hopper-medium-v2**
2. **walker2d-medium-v2**
3. **halfcheetah-medium-v2**

Metrics: `normalized_score`, `episode_reward`, `training_time`. Final scores use a geometric mean over the three environment-specific normalized-score terms.

## Baselines

### default — Diffusion Q-Learning (DQL)
The unmodified template ports CleanDiffuser's `dql_d4rl_mujoco.py` line-for-line: diffusion actor + twin Q critic, BC + Q loss. Reference: Wang, Hunt, Zhou, "Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning", ICLR 2023 (arXiv:2208.06193).

### idql — Implicit Diffusion Q-Learning
Decoupled actor / critic with τ-expectile IQL critic and softmax(adv * β) action reweighting at inference. Reference: Hansen-Estruch et al., 2023 (arXiv:2304.10573); built on IQL (Kostrikov, Nair, Levine, ICLR 2022, arXiv:2110.06169).

### diffusion_policy — Diffusion Policy
Pure behavior cloning with a diffusion actor (no critic, single-action sampling at inference). Reference: Chi et al., RSS 2023 / IJRR 2024 (arXiv:2303.04137).

## Evaluation Protocol

To keep the protocol fixed across all baselines / agents:

- `gradient_steps = 1,000,000` for every method. CleanDiffuser's package configs (`configs/{dql,idql}/mujoco/mujoco.yaml`) and the DQL paper both use 2M, but at 1M the DQL default already reproduces CleanDiffuser's published numbers, so 1M is the better walltime / quality tradeoff for this benchmark. The model may shorten training inside its own edits but cannot lengthen it.
- `num_candidates = 50` at inference for every method that uses Q-reranking (DQL, IDQL). This matches CleanDiffuser's DQL reference repro and the "DQL+selection" variant; CleanDiffuser's IDQL package config defaults to 256 but is reduced to 50 here so all reranking baselines see the same compute. `diffusion_policy` ignores `num_candidates` (sample-1 inference, no critic).
- `num_envs = 50`, `num_episodes = 3`, `use_ema = True` at inference.

Seed=42 is the primary seed; multi-seed (123, 456) is run for the `default` baseline to confirm hopper-medium-v2 is not cherry-picked.
