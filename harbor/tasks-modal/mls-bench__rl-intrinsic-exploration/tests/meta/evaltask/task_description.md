# RL Intrinsic Exploration: Sparse-Reward Novelty Bonus Design

## Research Question
Design an intrinsic exploration mechanism that improves sparse-reward
discovery in hard-exploration Atari environments.

## Background
In sparse-reward reinforcement learning, external rewards arrive too
infrequently for vanilla policy optimization to learn efficiently. A
common solution is to add an **intrinsic reward** that encourages
novelty, surprise, or state-space coverage on top of the (possibly
clipped) extrinsic reward.

This task isolates that question. The PPO training loop, Atari
preprocessing (grayscale, frame-skip, frame-stack, terminal-on-life-loss),
policy/value architecture, and optimizer are fixed. The only thing you
should redesign is the intrinsic-bonus module and how its signal is mixed
into learning.

Reference families include:
- **No bonus / vanilla PPO** — Schulman et al., "Proximal Policy
  Optimization Algorithms" (arXiv:1707.06347). Learns only from clipped
  extrinsic reward.
- **RND** — Burda et al., "Exploration by Random Network Distillation"
  (arXiv:1810.12894, ICLR 2019). Bonus is the prediction error of a
  learned network against a fixed randomly-initialized target network.
- **ICM** — Pathak et al., "Curiosity-driven Exploration by
  Self-supervised Prediction" (arXiv:1705.05363, ICML 2017). Bonus is the
  forward-dynamics prediction error in a feature space learned by an
  inverse-dynamics model.

## Editable Interface
You will modify the editable section of `custom_intrinsic_exploration.py`:
- `IntrinsicBonusModule` — defines how intrinsic rewards are computed and
  trained.
- `mix_advantages(...)` — defines how extrinsic and intrinsic advantages
  are combined.

The editable code must keep the public interface intact:
- `initialize(envs)`
- `trainable_parameters()`
- `update_batch_stats(batch_obs, batch_next_obs)`
- `compute_bonus(obs, next_obs, actions)`
- `normalize_rollout_rewards(rollout_intrinsic)`
- `loss(batch_obs, batch_next_obs, batch_actions)`
- `mix_advantages(ext_advantages, int_advantages, args)`

## Evaluation
The agent is trained with the same fixed PPO-style loop on multiple
sparse-reward Atari environments, including:
- **Tutankham-v5** — medium-difficulty visible game.
- **Frostbite-v5** — hard-exploration visible game.
- **PrivateEye-v5** — additional hard-exploration setting.

Reported metrics:
- `eval_return` — mean evaluation episodic return at the fixed training
  budget.
- `auc` — area under the evaluation-return curve across training.
- `nonzero_rate` — fraction of evaluation episodes with non-zero
  episodic return.

Evaluation uses deterministic rollouts with a fixed per-episode step
cap so that non-terminating Atari behavior cannot stall the benchmark.
Higher is better for all metrics. A method should improve across
multiple games rather than helping only one.
