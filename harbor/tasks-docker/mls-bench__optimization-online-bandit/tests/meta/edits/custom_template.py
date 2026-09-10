# Custom online bandit algorithm for MLS-Bench
#
# EDITABLE section: BanditPolicy class — the exploration-exploitation strategy.
# FIXED sections: everything else (environments, evaluation, main loop).
#
# Three evaluation settings:
#   1. Stochastic MAB (K=10 Bernoulli arms, T=10000)
#   2. Contextual Bandits (d=10 context, K=5 linear arms, T=10000)
#   3. Non-stationary MAB (K=5 Bernoulli arms with 4 abrupt changes, T=10000)
#
# Metric: normalized cumulative regret at horizon T (lower is better).

import argparse
import math
import os
import sys
from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import brentq
from scipy.special import rel_entr


# =====================================================================
# FIXED: Arm Distributions
# =====================================================================
class Arm(ABC):
    """Abstract arm with a mean reward and a draw method."""

    @abstractmethod
    def mean(self) -> float:
        ...

    @abstractmethod
    def draw(self, rng: np.random.Generator) -> float:
        ...


class BernoulliArm(Arm):
    """Bernoulli arm with parameter p in [0, 1]."""

    def __init__(self, p: float):
        assert 0.0 <= p <= 1.0
        self._p = p

    def mean(self) -> float:
        return self._p

    def draw(self, rng: np.random.Generator) -> float:
        return float(rng.random() < self._p)


class GaussianArm(Arm):
    """Gaussian arm with mean mu and std sigma (rewards clipped to [0, 1])."""

    def __init__(self, mu: float, sigma: float = 0.25):
        self._mu = mu
        self._sigma = sigma

    def mean(self) -> float:
        return self._mu

    def draw(self, rng: np.random.Generator) -> float:
        return float(np.clip(rng.normal(self._mu, self._sigma), 0.0, 1.0))


# =====================================================================
# FIXED: Bandit Environments
# =====================================================================
class StochasticMAB:
    """Standard stochastic multi-armed bandit with K Bernoulli arms.

    Arms have fixed, unknown reward probabilities.  The optimal policy
    always plays the arm with the highest mean.
    """

    def __init__(self, arm_means: list[float], seed: int = 42):
        self.arms = [BernoulliArm(p) for p in arm_means]
        self.K = len(self.arms)
        self.best_mean = max(a.mean() for a in self.arms)
        self.rng = np.random.default_rng(seed)
        self.context_dim = 0  # no context

    def reset(self):
        """Called at the start of each episode (no-op for stationary)."""
        pass

    def get_context(self) -> np.ndarray | None:
        """Return context vector, or None for context-free bandits."""
        return None

    def pull(self, arm: int) -> tuple[float, float]:
        """Pull an arm.  Returns (reward, instantaneous_regret)."""
        reward = self.arms[arm].draw(self.rng)
        regret = self.best_mean - self.arms[arm].mean()
        return reward, regret


class ContextualBandit:
    """Linear contextual bandit: reward = context^T theta_arm + noise.

    Each arm has a fixed (unknown) parameter vector theta_arm of dimension d.
    At each round, a context x is drawn uniformly from the unit sphere.
    Expected reward for arm a given context x is x^T theta_a.
    Noise is Gaussian with std=0.1, rewards clipped to [0, 1].
    """

    def __init__(self, K: int, d: int, seed: int = 42):
        self.K = K
        self.d = d
        self.context_dim = d
        self.rng = np.random.default_rng(seed)
        # Generate arm parameters: theta_a are unit-norm vectors
        raw = self.rng.standard_normal((K, d))
        self.theta = raw / np.linalg.norm(raw, axis=1, keepdims=True) * 0.5
        self._noise_std = 0.1
        self._current_context: np.ndarray | None = None

    def reset(self):
        self._current_context = None

    def get_context(self) -> np.ndarray:
        """Draw a fresh context vector from the unit sphere."""
        x = self.rng.standard_normal(self.d)
        x = x / np.linalg.norm(x)
        self._current_context = x
        return x.copy()

    def pull(self, arm: int) -> tuple[float, float]:
        """Pull an arm given the current context."""
        assert self._current_context is not None, "Call get_context() first"
        x = self._current_context
        expected_rewards = self.theta @ x  # (K,)
        best_reward = expected_rewards.max()
        arm_reward = expected_rewards[arm]
        noise = self.rng.normal(0, self._noise_std)
        reward = float(np.clip(arm_reward + noise, 0.0, 1.0))
        regret = best_reward - arm_reward
        return reward, regret


class NonStationaryMAB:
    """Piece-wise stationary MAB with abrupt changepoints.

    The arm means change at pre-specified timesteps.  A good algorithm
    must detect or adapt to these changes.
    """

    def __init__(
        self,
        arm_configs: list[list[float]],
        changepoints: list[int],
        seed: int = 42,
    ):
        """
        Args:
            arm_configs: list of arm-mean vectors, one per segment.
                         arm_configs[i] gives the K arm means for segment i.
            changepoints: sorted list of timesteps where the means change.
                          len(changepoints) == len(arm_configs) - 1.
        """
        assert len(arm_configs) == len(changepoints) + 1
        self.arm_configs = [np.array(c) for c in arm_configs]
        self.changepoints = changepoints
        self.K = len(arm_configs[0])
        self.context_dim = 0
        self.rng = np.random.default_rng(seed)
        self._t = 0
        self._segment = 0

    def reset(self):
        self._t = 0
        self._segment = 0

    def get_context(self) -> np.ndarray | None:
        return None

    def pull(self, arm: int) -> tuple[float, float]:
        # Advance segment if needed
        while (
            self._segment < len(self.changepoints)
            and self._t >= self.changepoints[self._segment]
        ):
            self._segment += 1
        means = self.arm_configs[self._segment]
        best_mean = means.max()
        arm_mean = means[arm]
        reward = float(self.rng.random() < arm_mean)  # Bernoulli
        regret = best_mean - arm_mean
        self._t += 1
        return reward, regret


# =====================================================================
# FIXED: Environment Factory
# =====================================================================
def make_env(env_name: str, seed: int):
    """Create a bandit environment by name.

    Returns (env, horizon) where horizon is the number of rounds.
    """
    if env_name == "stochastic_mab":
        # 10-armed Bernoulli bandit
        arm_means = [0.10, 0.20, 0.30, 0.35, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80]
        return StochasticMAB(arm_means, seed=seed), 10000

    elif env_name == "contextual":
        # 5-armed linear contextual bandit, d=10
        return ContextualBandit(K=5, d=10, seed=seed), 10000

    elif env_name == "nonstationary":
        # 5-armed piece-wise stationary with 4 changepoints
        configs = [
            [0.9, 0.3, 0.2, 0.1, 0.5],   # segment 0: arm 0 best
            [0.2, 0.8, 0.3, 0.1, 0.4],   # segment 1: arm 1 best
            [0.1, 0.2, 0.7, 0.3, 0.5],   # segment 2: arm 2 best
            [0.3, 0.1, 0.2, 0.85, 0.4],  # segment 3: arm 3 best
            [0.2, 0.4, 0.3, 0.1, 0.9],   # segment 4: arm 4 best
        ]
        changepoints = [2000, 4000, 6000, 8000]
        return NonStationaryMAB(configs, changepoints, seed=seed), 10000

    else:
        raise ValueError(f"Unknown environment: {env_name}")


# =====================================================================
# FIXED: KL-divergence utilities (for reference — usable by the agent)
# =====================================================================
def kl_bernoulli(p: float, q: float) -> float:
    """KL(Bernoulli(p) || Bernoulli(q)), with safe handling of edge cases."""
    p = np.clip(p, 1e-10, 1 - 1e-10)
    q = np.clip(q, 1e-10, 1 - 1e-10)
    return float(rel_entr(p, q) + rel_entr(1 - p, 1 - q))


def kl_ucb_bound(mu_hat: float, n: int, t: int, c: float = 1.0) -> float:
    """Compute the KL-UCB upper confidence bound for a Bernoulli arm.

    Finds max q in [mu_hat, 1] such that n * KL(mu_hat, q) <= c * log(t).
    """
    if n == 0:
        return 1.0
    mu_hat = np.clip(mu_hat, 1e-10, 1 - 1e-10)
    threshold = c * math.log(max(t, 1)) / n

    def f(q):
        return kl_bernoulli(mu_hat, q) - threshold

    if f(1.0 - 1e-10) <= 0:
        return 1.0
    try:
        return brentq(f, mu_hat, 1.0 - 1e-10, xtol=1e-6)
    except ValueError:
        return 1.0


# =====================================================================
# EDITABLE: BanditPolicy
# =====================================================================
class BanditPolicy:
    """Bandit policy: the agent's exploration-exploitation strategy.

    The evaluation loop calls:
        policy = BanditPolicy(K, context_dim)
        policy.reset()
        for t in range(T):
            context = env.get_context()          # None for MAB
            arm = policy.select_arm(t, context)  # choose arm
            reward, _ = env.pull(arm)
            policy.update(arm, reward, context)  # observe reward

    You MUST implement:
        select_arm(t, context) -> int   : pick an arm in {0, ..., K-1}
        update(arm, reward, context)    : update internal state
        reset()                         : reset state for a new run

    Available utilities (fixed, importable):
        kl_bernoulli(p, q)              : KL divergence between Bernoulli(p) and Bernoulli(q)
        kl_ucb_bound(mu_hat, n, t, c)   : KL-UCB upper confidence bound

    Args:
        K: number of arms
        context_dim: dimension of context vector (0 if no context)
    """

    def __init__(self, K: int, context_dim: int = 0):
        self.K = K
        self.context_dim = context_dim
        self.counts = np.zeros(K, dtype=np.float64)
        self.rewards = np.zeros(K, dtype=np.float64)

    def reset(self):
        """Reset internal state for a new run."""
        self.counts[:] = 0
        self.rewards[:] = 0

    def select_arm(self, t: int, context: np.ndarray | None = None) -> int:
        """Select which arm to pull at timestep t.

        Args:
            t: current timestep (0-indexed)
            context: context vector of shape (context_dim,), or None

        Returns:
            arm index in {0, ..., K-1}
        """
        # Placeholder: uniform random — replace with your algorithm
        return int(np.random.randint(self.K))

    def update(self, arm: int, reward: float, context: np.ndarray | None = None):
        """Update internal state after observing a reward.

        Args:
            arm: the arm that was pulled
            reward: the observed reward
            context: the context vector that was active, or None
        """
        self.counts[arm] += 1
        self.rewards[arm] += reward


# =====================================================================
# FIXED: Evaluation Protocol
# =====================================================================
def run_bandit(env, policy, horizon: int) -> dict:
    """Run a bandit algorithm for `horizon` steps.

    Returns dict with cumulative_regret, normalized_regret, and per-step info.
    """
    env.reset()
    policy.reset()

    cumulative_regret = 0.0
    regret_history = []

    for t in range(horizon):
        context = env.get_context()
        arm = policy.select_arm(t, context)
        reward, regret = env.pull(arm)
        policy.update(arm, reward, context)
        cumulative_regret += regret
        if (t + 1) % 1000 == 0:
            regret_history.append((t + 1, cumulative_regret))

    # Normalized regret: cumulative_regret / horizon
    normalized_regret = cumulative_regret / horizon

    return {
        "cumulative_regret": cumulative_regret,
        "normalized_regret": normalized_regret,
        "regret_history": regret_history,
    }


def evaluate(env_name: str, seed: int, output_dir: str | None = None):
    """Evaluate the BanditPolicy on a given environment."""
    env, horizon = make_env(env_name, seed=seed)
    policy = BanditPolicy(K=env.K, context_dim=env.context_dim)
    result = run_bandit(env, policy, horizon)

    # Print training progress
    for step, cum_reg in result["regret_history"]:
        norm_reg = cum_reg / step
        print(
            f"TRAIN_METRICS step={step} cumulative_regret={cum_reg:.4f} "
            f"normalized_regret={norm_reg:.6f}",
            flush=True,
        )

    # Print final test metrics
    print(
        f"TEST_METRICS cumulative_regret={result['cumulative_regret']:.4f} "
        f"normalized_regret={result['normalized_regret']:.6f}",
        flush=True,
    )

    return result


# =====================================================================
# FIXED: Main
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Online Bandit Evaluation")
    parser.add_argument("--env", type=str, required=True,
                        choices=["stochastic_mab", "contextual", "nonstationary"],
                        help="Bandit environment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (optional)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    evaluate(args.env, seed=args.seed, output_dir=args.output_dir)
