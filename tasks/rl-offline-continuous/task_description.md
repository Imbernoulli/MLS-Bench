# Offline RL: Q-Value Overestimation Suppression in Continuous Control

## Research Question
Design and implement an offline RL algorithm that suppresses Q-value
overestimation while learning useful policies from a static dataset.
Your code goes in `custom.py`. Several reference implementations are
provided as read-only `*.edit.py` baselines.

## Background
In offline RL the agent cannot collect new transitions, so standard
bootstrapped Q-learning tends to overestimate values for
out-of-distribution actions, which then drive the policy away from the
data and degrade performance. Different mechanisms — conservative value
penalties, behavior regularization, expectile-style value functions,
ensemble pessimism — trade off in-distribution exploitation against
out-of-distribution caution.

Reference baselines spanning the design space:
- **ReBRAC** — Tarasov et al., "Revisiting the Minimalist Approach to
  Offline Reinforcement Learning" (arXiv:2305.09836, NeurIPS 2023).
  Decoupled actor and critic BC penalties on top of TD3+BC.
- **TD3+BC** — Fujimoto and Gu, "A Minimalist Approach to Offline
  Reinforcement Learning" (arXiv:2106.06860, NeurIPS 2021). TD3
  augmented with a normalized BC term in the actor loss with default
  coefficient `alpha = 2.5`.
- **IQL** — Kostrikov et al., "Offline Reinforcement Learning with
  Implicit Q-Learning" (arXiv:2110.06169, ICLR 2022). Expectile
  regression with default `tau = 0.7` and advantage-weighted policy
  extraction temperature `beta = 3.0` for D4RL MuJoCo.

## Constraints
- **Network dimensions are fixed at 256.** All MLP hidden layers must
  use 256 units. A `_mlp()` factory function is provided in the FIXED
  section for convenience. You may define custom network classes but
  hidden widths must remain 256.
- **Total parameter count is enforced.** The training loop checks that
  total trainable parameters do not exceed 1.2x the largest baseline
  architecture, so the contribution must be algorithmic (losses,
  regularization, target construction, policy extraction) rather than
  capacity.
- Do **not** simply copy a reference implementation with minor changes.

## Evaluation
Trained and evaluated on D4RL MuJoCo continuous-control datasets
including HalfCheetah, Hopper and Walker2d using `medium-v2` data.
Metric: D4RL normalized score (0 = random, 100 = expert), averaged over
evaluation rollouts. Higher is better. Strong methods should generalize
across the locomotion datasets rather than relying on dataset-specific
quirks.
