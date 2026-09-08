"""Thompson Sampling baseline for opt-online-bandit.

Bayesian approach to exploration-exploitation.  Maintains a Beta posterior
for each arm's success probability (for Bernoulli rewards):
    theta_a ~ Beta(alpha_a, beta_a)

At each round, samples theta_a from each arm's posterior and plays the
arm with the highest sample.  For contextual bandits, uses Thompson
Sampling with a Gaussian linear model (LinTS) with Sherman-Morrison
incremental updates for efficiency.

For non-stationary settings, uses a discounted posterior that gives
more weight to recent observations.

Reference:
  Thompson. "On the Likelihood that One Unknown Probability Exceeds
  Another in View of the Evidence of Two Samples." Biometrika, 1933.

  Agrawal & Goyal. "Analysis of Thompson Sampling for the Multi-armed
  Bandit Problem." COLT 2012.

  Agrawal & Goyal. "Thompson Sampling for Contextual Bandits with
  Linear Payoffs." ICML 2013. (LinTS)

Reference code: vendor/external_packages/SMPyBandits/SMPyBandits/Policies/Thompson.py
"""

_FILE = "SMPyBandits/custom_bandit.py"

_TS_CODE = """\
class BanditPolicy:
    \"\"\"Thompson Sampling with Beta posterior for Bernoulli arms.

    For MAB: samples from Beta(alpha, beta) posterior per arm.
    For contextual bandits: uses Bayesian linear regression (LinTS)
    with Sherman-Morrison incremental inverse updates.
    For non-stationary: uses discounted posterior (gamma < 1).
    \"\"\"

    def __init__(self, K: int, context_dim: int = 0):
        self.K = K
        self.context_dim = context_dim
        self.rng = np.random.default_rng(np.random.randint(0, 2**32 - 1))

        # Beta posterior params for MAB (alpha=successes+1, beta=failures+1)
        self.alpha = np.ones(K, dtype=np.float64)
        self.beta_param = np.ones(K, dtype=np.float64)

        # Discount factor for non-stationary settings
        self._gamma = 0.999

        # LinTS parameters for contextual bandits
        if context_dim > 0:
            self._lambda = 1.0  # regularization
            self._v2 = 0.25  # sampling variance scale
            # B_inv_a via Sherman-Morrison updates
            self._B_inv = np.array([np.eye(context_dim) / self._lambda
                                    for _ in range(K)])
            self._f = np.zeros((K, context_dim), dtype=np.float64)
            self._theta_hat = np.zeros((K, context_dim), dtype=np.float64)

        # Tracking
        self.counts = np.zeros(K, dtype=np.float64)
        self.rewards = np.zeros(K, dtype=np.float64)

    def reset(self):
        self.alpha[:] = 1.0
        self.beta_param[:] = 1.0
        self.counts[:] = 0
        self.rewards[:] = 0
        if self.context_dim > 0:
            d = self.context_dim
            for a in range(self.K):
                self._B_inv[a] = np.eye(d) / self._lambda
                self._f[a] = np.zeros(d)
                self._theta_hat[a] = np.zeros(d)

    def select_arm(self, t: int, context: np.ndarray | None = None) -> int:
        if context is not None and self.context_dim > 0:
            return self._lints_select(context)

        # Sample from Beta posterior for each arm
        samples = self.rng.beta(self.alpha, self.beta_param)
        return int(np.argmax(samples))

    def _lints_select(self, context: np.ndarray) -> int:
        \"\"\"Linear Thompson Sampling for contextual bandits.\"\"\"
        best_arm = 0
        best_val = -np.inf
        for a in range(self.K):
            mu_a = self._theta_hat[a]
            # Sample: theta ~ N(mu_a, v2 * B_inv_a)
            # Use Cholesky of B_inv for efficient sampling
            z = self.rng.standard_normal(self.context_dim)
            try:
                L = np.linalg.cholesky(self._v2 * self._B_inv[a])
                theta_sample = mu_a + L @ z
            except np.linalg.LinAlgError:
                theta_sample = mu_a + math.sqrt(self._v2) * z
            val = context @ theta_sample
            if val > best_val:
                best_val = val
                best_arm = a
        return best_arm

    def update(self, arm: int, reward: float, context: np.ndarray | None = None):
        self.counts[arm] += 1
        self.rewards[arm] += reward

        if context is not None and self.context_dim > 0:
            # Sherman-Morrison update: B_inv -= (B_inv x x^T B_inv)/(1 + x^T B_inv x)
            Bx = self._B_inv[arm] @ context
            denom = 1.0 + context @ Bx
            self._B_inv[arm] -= np.outer(Bx, Bx) / denom
            self._f[arm] += reward * context
            self._theta_hat[arm] = self._B_inv[arm] @ self._f[arm]
        else:
            # Discounted Beta posterior update (for non-stationary robustness)
            self.alpha *= self._gamma
            self.beta_param *= self._gamma
            # Clamp to prevent posterior from collapsing
            self.alpha = np.maximum(self.alpha, 1.0)
            self.beta_param = np.maximum(self.beta_param, 1.0)
            # Update the pulled arm
            self.alpha[arm] += reward
            self.beta_param[arm] += (1.0 - reward)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 261,
        "end_line": 321,
        "content": _TS_CODE,
    },
]
