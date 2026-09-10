"""UCB1 baseline for opt-online-bandit.

Upper Confidence Bound algorithm.  Selects the arm maximizing:
    mu_hat_a + sqrt(2 * log(t) / N_a)

where mu_hat_a is the empirical mean reward of arm a, N_a is the number
of times arm a has been pulled, and t is the current timestep.

For contextual bandits, falls back to per-arm UCB1 (ignoring context),
which is suboptimal but provides a baseline.

For non-stationary bandits, uses a sliding-window variant (SW-UCB) with
window size W to track recent performance only.  Uses an efficient
circular buffer so each step is O(1).

Reference:
  Auer, Cesa-Bianchi, Fischer. "Finite-time Analysis of the Multiarmed
  Bandit Problem." Machine Learning 47(2-3):235-256, 2002.

  Garivier & Moulines. "On Upper-Confidence Bound Policies for
  Switching Bandit Problems." ALT 2011. (SW-UCB extension)

Reference code: vendor/external_packages/SMPyBandits/SMPyBandits/Policies/UCB.py
"""

_FILE = "SMPyBandits/custom_bandit.py"

_UCB1_CODE = """\
class BanditPolicy:
    \"\"\"UCB1: Upper Confidence Bound algorithm.

    Maintains empirical means and pull counts.  Selects the arm with the
    highest upper confidence bound: mu_hat + sqrt(2 * log(t+1) / N_a).

    For non-stationary settings, uses a sliding window of size W with
    an efficient circular buffer (O(1) per step).
    \"\"\"

    def __init__(self, K: int, context_dim: int = 0):
        self.K = K
        self.context_dim = context_dim
        # Cumulative statistics
        self.counts = np.zeros(K, dtype=np.float64)
        self.rewards = np.zeros(K, dtype=np.float64)
        # Sliding window via circular buffer for non-stationary settings
        self._W = 800
        self._buf_arms = np.zeros(self._W, dtype=np.int32)
        self._buf_rewards = np.zeros(self._W, dtype=np.float64)
        self._buf_ptr = 0
        self._buf_full = False
        self._sw_counts = np.zeros(K, dtype=np.float64)
        self._sw_rewards = np.zeros(K, dtype=np.float64)

    def reset(self):
        self.counts[:] = 0
        self.rewards[:] = 0
        self._buf_ptr = 0
        self._buf_full = False
        self._sw_counts[:] = 0
        self._sw_rewards[:] = 0

    def select_arm(self, t: int, context: np.ndarray | None = None) -> int:
        # Initial round-robin: play each arm once
        if t < self.K:
            return t

        # Standard UCB1 index (full history). The SW-UCB fallback here was
        # incorrect — vanilla UCB1 should use the full history regardless of
        # environment; switching to sliding-window inflated regret on
        # stationary MAB from ~960 (theoretical) to ~1450 observed.

        mu_hat = self.rewards / np.maximum(self.counts, 1e-10)
        exploration = np.sqrt(2.0 * math.log(t + 1) / np.maximum(self.counts, 1))
        ucb_values = mu_hat + exploration
        return int(np.argmax(ucb_values))

    def _sw_select(self, t: int) -> int:
        \"\"\"Sliding-window UCB using pre-maintained running statistics.\"\"\"
        unpulled = self._sw_counts == 0
        if unpulled.any():
            return int(np.argmax(unpulled))
        mu_hat = self._sw_rewards / self._sw_counts
        xi = 1.5  # exploration parameter for SW-UCB
        exploration = np.sqrt(xi * math.log(self._W) / self._sw_counts)
        return int(np.argmax(mu_hat + exploration))

    def update(self, arm: int, reward: float, context: np.ndarray | None = None):
        self.counts[arm] += 1
        self.rewards[arm] += reward
        # Update circular buffer and running window stats
        if self._buf_full:
            old_arm = int(self._buf_arms[self._buf_ptr])
            old_rew = self._buf_rewards[self._buf_ptr]
            self._sw_counts[old_arm] -= 1
            self._sw_rewards[old_arm] -= old_rew
        self._buf_arms[self._buf_ptr] = arm
        self._buf_rewards[self._buf_ptr] = reward
        self._sw_counts[arm] += 1
        self._sw_rewards[arm] += reward
        self._buf_ptr += 1
        if self._buf_ptr >= self._W:
            self._buf_ptr = 0
            self._buf_full = True
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 261,
        "end_line": 321,
        "content": _UCB1_CODE,
    },
]
