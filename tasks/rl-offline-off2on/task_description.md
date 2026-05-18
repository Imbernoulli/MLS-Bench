# Offline-to-Online RL: Preventing Catastrophic Forgetting in Fine-Tuning

## Research Question
Design and implement an offline-to-online RL algorithm that pretrains
from an offline dataset and then fine-tunes with online interaction
without catastrophic forgetting or Q-value collapse. Your code goes in
`custom_finetune.py`. Several reference implementations are provided as
read-only `*.edit.py` baselines.

## Background
Offline-to-online RL pretrains a policy and value function on a fixed
dataset and then continues learning with environment interaction. The
offline-to-online transition is brittle: conservative offline value
functions can become overoptimistic once online data shifts the replay
distribution, behavior-regularized policies can forget useful offline
behavior, and naive fine-tuning often causes a Q-value collapse and a
performance drop early in the online phase.

The Adroit `cloned-v1` datasets mix expert and noisy demonstrations, so
the offline pretraining never produces a strong policy on its own and
the online phase must improve substantially without losing what little
competence was learned.

Reference baselines spanning the design space:
- **AWAC** — Nair et al., "AWAC: Accelerating Online Reinforcement
  Learning with Offline Datasets" (arXiv:2006.09359). Implicit
  advantage-weighted policy constraint that allows smooth fine-tuning.
  Default Lagrange temperature `lambda = 1.0`.
- **SPOT** — Wu et al., "Supported Policy Optimization for Offline
  Reinforcement Learning" (arXiv:2202.06239, NeurIPS 2022). VAE-based
  density support constraint that supports online fine-tuning.
- **IQL** — Kostrikov et al., "Offline Reinforcement Learning with
  Implicit Q-Learning" (arXiv:2110.06169, ICLR 2022). Expectile
  regression pretraining with advantage-weighted policy extraction,
  providing a stable offline initialization for online fine-tuning.

## Constraints
- **Network dimensions are fixed at 256.** All MLP hidden layers must
  use 256 units. A `_mlp()` factory function is provided in the FIXED
  section for convenience. You may define custom network classes but
  hidden widths must remain 256.
- **Total parameter count is enforced.** The training loop checks that
  total trainable parameters do not exceed 1.2x the largest baseline
  architecture, so the contribution must be algorithmic (transition
  handling, value calibration, replay balancing, behavior-constraint
  annealing) rather than capacity.
- Do **not** simply copy a reference implementation with minor changes.

## Evaluation
Trained and evaluated on Adroit tasks Pen, Door and Hammer using the
D4RL `cloned-v1` datasets. The offline phase trains for 1M gradient
steps, then the online phase trains for 1M environment-interaction
steps. Metric: D4RL normalized score (0 = random, 100 = expert),
evaluated throughout both phases. Higher is better. Strong methods
should retain offline competence while benefiting from online
fine-tuning across the manipulation tasks.
