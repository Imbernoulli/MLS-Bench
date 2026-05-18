# Offline RL: Dexterous Manipulation with Narrow Expert Data (Adroit)

## Research Question
Design and implement an offline RL algorithm for high-dimensional
dexterous manipulation from narrow human-demonstration data. Your code
goes in `custom_adroit.py`. Several reference implementations are
provided as read-only `*.edit.py` baselines.

## Background
Adroit tasks involve a 24-DoF simulated robotic hand with high-dimensional
action spaces (24–30 dims). The D4RL `human-v1` datasets contain only
roughly 25 human teleoperation demonstrations per task, creating severe
distribution shift compared to locomotion-style offline RL. Standard
Q-learning on this data tends to extrapolate badly outside the narrow
data support, while pure behavior cloning is limited by the small
dataset.

Reference baselines spanning the design space:
- **IQL** — Kostrikov et al., "Offline Reinforcement Learning with
  Implicit Q-Learning" (arXiv:2110.06169, ICLR 2022). Expectile
  regression with advantage-weighted policy extraction; well-suited to
  narrow data support without requiring OOD action queries.
- **AWAC** — Nair et al., "AWAC: Accelerating Online Reinforcement
  Learning with Offline Datasets" (arXiv:2006.09359). Advantage-weighted
  actor-critic with implicit policy constraint (default temperature
  `lambda = 1.0` per paper).
- **ReBRAC** — Tarasov et al., "Revisiting the Minimalist Approach to
  Offline Reinforcement Learning" (arXiv:2305.09836, NeurIPS 2023).
  TD3+BC-style actor-critic with decoupled actor / critic BC penalties;
  per-domain BC coefficients are tuned per the paper's appendix.

## Constraints
- **Network dimensions are fixed at 256.** All MLP hidden layers must
  use 256 units. A `_mlp()` factory function is provided in the FIXED
  section for convenience. You may define custom network classes but
  hidden widths must remain 256.
- **Total parameter count is enforced.** The training loop checks that
  total trainable parameters do not exceed 1.2x the largest baseline
  architecture, so the contribution must be algorithmic (loss,
  regularization, target construction, training procedure) rather than
  capacity.
- Do **not** simply copy a reference implementation with minor changes.

## Evaluation
Trained and evaluated on Adroit tasks Pen (rotation), Door (opening) and
Hammer (nailing) using the D4RL `human-v1` datasets. Metric: D4RL
normalized score (0 = random performance, 100 = expert), averaged over
evaluation rollouts. Higher is better. Strong methods should work across
the manipulation tasks rather than overfitting to a single dataset.
