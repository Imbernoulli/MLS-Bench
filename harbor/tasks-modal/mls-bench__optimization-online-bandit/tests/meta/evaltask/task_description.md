# Online Bandits: Exploration-Exploitation Strategy Design

## Objective
Design and implement a bandit policy that minimizes cumulative regret across diverse multi-armed bandit settings. Your code goes in `custom_bandit.py`. Three reference implementations (UCB1, Thompson Sampling, KL-UCB) are available as read-only in the SMPyBandits package.

## Background
The multi-armed bandit problem is a fundamental model for the exploration-exploitation tradeoff in sequential decision-making. At each round, an agent selects one of K arms and observes a stochastic reward. The goal is to minimize cumulative regret — the gap between the reward of the best arm (in hindsight) and the agent's actual reward.

Classic algorithms include:
- **UCB1** (Auer, Cesa-Bianchi, and Fischer, "Finite-time Analysis of the Multiarmed Bandit Problem", *Machine Learning* 47, 2002): plays the arm with the highest upper confidence bound `mu_hat + sqrt(2 log(t) / n_a)`, achieving `O(sqrt(KT log T))` minimax regret.
- **Thompson Sampling** (Thompson, 1933; Agrawal and Goyal, "Analysis of Thompson Sampling for the Multi-armed Bandit Problem", COLT 2012): samples from a Bayesian posterior and plays the arm with the highest sample, achieving optimal Bayesian regret.
- **KL-UCB** (Garivier and Cappé, "The KL-UCB Algorithm for Bounded Stochastic Bandits and Beyond", COLT 2011; arXiv:1102.2490; Cappé, Garivier, Maillard, Munos, and Stoltz, *Annals of Statistics* 41, 2013): uses Kullback-Leibler divergence for tighter confidence bounds, provably optimal for Bernoulli bandits.

Key challenges include adapting to different reward distributions, handling contextual information, and detecting non-stationarity.

## Task
Modify the `BanditPolicy` class in `custom_bandit.py` (the EDITABLE section). You must implement:
- `__init__(K, context_dim)`: initialize your policy for K arms with optional context.
- `select_arm(t, context)`: choose which arm to pull at timestep t.
- `update(arm, reward, context)`: update internal state after observing a reward.
- `reset()`: reset state for a new run.

## Interface
```python
class BanditPolicy:
    def __init__(self, K: int, context_dim: int = 0): ...
    def reset(self): ...
    def select_arm(self, t: int, context: np.ndarray | None = None) -> int: ...
    def update(self, arm: int, reward: float, context: np.ndarray | None = None): ...
```

Available utilities (in the FIXED section):
- `kl_bernoulli(p, q)`: KL divergence between Bernoulli distributions.
- `kl_ucb_bound(mu_hat, n, t, c)`: computes the KL-UCB upper confidence bound (Garivier and Cappé, 2011).

## Evaluation
Evaluated on three bandit settings (lower regret is better):

1. **Stochastic MAB**: 10-armed Bernoulli bandit, T = 10,000 rounds. Arms have fixed reward probabilities.
2. **Contextual**: 5-armed linear contextual bandit with `d = 10` features, T = 10,000 rounds. Expected reward is a linear function of the context.
3. **Non-stationary**: 5-armed piece-wise stationary Bernoulli bandit with 4 abrupt changepoints, T = 10,000 rounds. The best arm changes over time.

Metric: normalized cumulative regret = `(cumulative regret) / T`.

## Baselines (paper-cited reference implementations from SMPyBandits)
- **ucb1** — Auer, Cesa-Bianchi, and Fischer (*Machine Learning* 2002); paper-default exploration constant `c = 2` in the `sqrt(c log t / n_a)` term.
- **thompson_sampling** — Thompson (1933) / Agrawal and Goyal (COLT 2012); paper-default `Beta(1, 1)` prior per arm for Bernoulli rewards.
- **kl_ucb** — Garivier and Cappé (COLT 2011; arXiv:1102.2490); paper-default exploration function `f(t) = log(t) + 3 log log(t)` and binary KL inversion via bisection.
