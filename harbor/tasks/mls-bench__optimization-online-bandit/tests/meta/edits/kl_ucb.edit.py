"""KL-UCB baseline for opt-online-bandit.

Kullback-Leibler Upper Confidence Bound algorithm.  Selects the arm
maximizing:
    max { q in [0,1] : N_a * KL(mu_hat_a, q) <= log(t) }

where KL is the Bernoulli KL divergence.  This is provably optimal for
Bernoulli bandits (matches the Lai-Robbins lower bound).

Uses a fast hand-written binary search for the KL-UCB bound computation
(avoids scipy.optimize.brentq overhead).

For contextual bandits, falls back to per-arm KL-UCB (ignoring context).
For non-stationary, uses a sliding-window variant with efficient O(1)
circular buffer updates.

Reference:
  Garivier & Cappe. "The KL-UCB Algorithm for Bounded Stochastic Bandits
  and Beyond." COLT 2011.

  Cappe, Garivier, Maillard, Munos, Stoltz. "Kullback-Leibler Upper
  Confidence Bounds for Optimal Sequential Allocation." Annals of
  Statistics 41(3):1516-1541, 2013.

Reference code: vendor/external_packages/SMPyBandits/SMPyBandits/Policies/klUCB.py
"""

_FILE = "SMPyBandits/custom_bandit.py"

_KLUCB_CODE = """\
class BanditPolicy:
    \"\"\"KL-UCB: Kullback-Leibler Upper Confidence Bound.

    Vanilla KL-UCB per Garivier & Cappe 2011.  Index for arm a at time t is:
        U_a(t) = sup { q in [0,1] : N_a(t) * kl(mu_hat_a, q) <= c*log(t) }
    with c = 1 (theorem-tight constant) and kl the Bernoulli KL divergence.

    Implements the Bernoulli KL-UCB index formula used by SMPyBandits; no
    sliding window (which harms the stationary stochastic regime).
    \"\"\"

    def __init__(self, K: int, context_dim: int = 0):
        self.K = K
        self.context_dim = context_dim
        self.counts = np.zeros(K, dtype=np.float64)
        self.rewards = np.zeros(K, dtype=np.float64)

    def reset(self):
        self.counts[:] = 0
        self.rewards[:] = 0

    @staticmethod
    def _fast_kl_ucb(p: float, n: int, t: int) -> float:
        \"\"\"Fast KL-UCB bound via binary search (no scipy dependency).\"\"\"
        if n == 0:
            return 1.0
        p = max(min(p, 1 - 1e-10), 1e-10)
        threshold = math.log(max(t, 1)) / n
        lo, hi = p, 1.0 - 1e-10
        for _ in range(32):  # 32 iterations gives ~1e-10 precision
            mid = (lo + hi) * 0.5
            # KL(Bernoulli(p) || Bernoulli(mid))
            kl = p * math.log(p / mid) + (1 - p) * math.log((1 - p) / (1 - mid))
            if kl < threshold:
                lo = mid
            else:
                hi = mid
        return (lo + hi) * 0.5

    def select_arm(self, t: int, context: np.ndarray | None = None) -> int:
        # Initial round-robin: each arm once.
        if t < self.K:
            return t

        # Standard KL-UCB index for each arm (no sliding window).
        best_arm = 0
        best_idx = -1e100
        for a in range(self.K):
            if self.counts[a] == 0:
                return a
            mu_hat = self.rewards[a] / self.counts[a]
            idx = self._fast_kl_ucb(mu_hat, int(self.counts[a]), t + 1)
            if idx > best_idx:
                best_idx = idx
                best_arm = a
        return best_arm

    def update(self, arm: int, reward: float, context: np.ndarray | None = None):
        self.counts[arm] += 1
        self.rewards[arm] += reward
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 261,
        "end_line": 321,
        "content": _KLUCB_CODE,
    },
]
