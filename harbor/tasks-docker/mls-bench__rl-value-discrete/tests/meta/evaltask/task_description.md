# Online RL: Value-Based Methods for Discrete Control

## Research Question
Design and implement a value-based RL algorithm for discrete action
spaces. Your code goes in `custom_value_discrete.py`. Several reference
implementations are provided as read-only `*.edit.py` baselines.

## Background
Value-based methods estimate Q-values `Q(s, a)` for each state-action
pair and derive a policy by acting greedily (or epsilon-greedily) with
respect to those estimates. In small to medium discrete-control tasks,
the key algorithmic challenges are overestimation bias under
bootstrapped targets, unstable value learning, exploration scheduling,
and representing uncertainty or full return distributions.

Reference baselines spanning the design space:
- **QR-DQN** — Dabney et al., "Distributional Reinforcement Learning
  with Quantile Regression" (arXiv:1710.10044, AAAI 2018).
  Quantile-regression distributional critic with default 200 quantiles
  trained with the Huber quantile loss.
- **Dueling DQN** — Wang et al., "Dueling Network Architectures for
  Deep Reinforcement Learning" (arXiv:1511.06581, ICML 2016). Splits
  the head into state value and action advantage streams.
- **C51** — Bellemare, Dabney and Munos, "A Distributional Perspective
  on Reinforcement Learning" (arXiv:1707.06887, ICML 2017). Categorical
  distributional critic with default 51 atoms over `[-10, 10]`.

## Constraints
- Network architecture dimensions are FIXED and cannot be modified.
- Total parameter count is enforced at runtime; the contribution must
  be algorithmic (head design, target construction, TD loss,
  exploration, replay usage) rather than encoder capacity.
- Do **not** simply copy a reference implementation with minor changes.

## Evaluation
Trained and evaluated on Gymnasium discrete-control tasks including
CartPole-v1, LunarLander-v2 and Acrobot-v1 within a fixed interaction
budget. Metric: mean episodic return over greedy evaluation episodes
(higher is better). Strong methods should remain stable across tasks
with different reward scales and episode lengths.
