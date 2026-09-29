# Custom online bandit algorithm for MLS-Bench
#
# EDITABLE section: BanditPolicy class — the exploration-exploitation strategy.
# FIXED sections: everything else (utilities and the worker loop below).
#
# Three evaluation settings (see mlsb_bandit_harness.py, read-only):
#   1. Stochastic MAB (K=10 Bernoulli arms, T=10000)
#   2. Contextual Bandits (d=10 context, K=5 linear arms, T=10000)
#   3. Non-stationary MAB (K=5 Bernoulli arms with 4 abrupt changes, T=10000)
#
# The environments live in the fixed harness, not in this file.  The harness
# runs this file as a separate process (`custom_bandit.py --worker`) and
# drives BanditPolicy over a pipe; every instance (arm means, parameters,
# changepoints, noise, contexts) is drawn from fresh OS entropy in the harness
# and is never visible to this process.  SEED seeds only np.random here.
#
# Metric: normalized cumulative regret at horizon T (lower is better).

import argparse
import math
import os
import sys

import numpy as np
from scipy.optimize import brentq
from scipy.special import rel_entr


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
# FIXED: Worker loop (driven by mlsb_bandit_harness.py over a pipe)
# =====================================================================
def _worker_main():
    """Serve BanditPolicy decisions to the harness.

    Protocol lines are written to the original stdout only; anything the
    policy prints goes to stderr.
    """
    proto_in = sys.stdin
    proto_out = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    policy = None
    prev_context = None
    for line in proto_in:
        parts = line.split()
        if not parts:
            continue
        cmd = parts[0]
        if cmd == "INIT":
            K, context_dim, _horizon, seed = (int(v) for v in parts[1:5])
            np.random.seed(seed)
            policy = BanditPolicy(K=K, context_dim=context_dim)
            policy.reset()
            prev_context = None
            proto_out.write("READY\n")
        elif cmd == "S":
            t, prev_arm = int(parts[1]), int(parts[2])
            prev_reward = float(parts[3])
            if prev_arm >= 0:
                policy.update(prev_arm, prev_reward, prev_context)
            context = (np.array([float(v) for v in parts[4:]])
                       if len(parts) > 4 else None)
            arm = policy.select_arm(t, None if context is None
                                    else context.copy())
            prev_context = context
            proto_out.write("%d\n" % int(arm))
        elif cmd == "U":
            policy.update(int(parts[1]), float(parts[2]), prev_context)
        elif cmd == "BYE":
            break
    proto_out.close()


# =====================================================================
# FIXED: Main
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Online bandit policy worker")
    parser.add_argument("--worker", action="store_true",
                        help="serve the harness protocol on stdin/stdout")
    args = parser.parse_args()
    if not args.worker:
        sys.exit("custom_bandit.py is driven by the harness; run e.g. "
                 "scripts/stochastic_mab.sh (python -I "
                 "SMPyBandits/mlsb_bandit_harness.py ...).")
    _worker_main()
