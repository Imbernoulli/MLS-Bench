"""Optuna CMA-ES baseline for opt-hyperparameter-search.

Reference: Hansen & Ostermeier (2001). "Completely Derandomized
Self-Adaptation in Evolution Strategies." Evolutionary Computation.
+ Nomura, Watanabe, Akimoto, Ozaki & Onishi (2021). "Warm Starting
CMA-ES for Hyperparameter Optimization." AAAI.

CMA-ES (Covariance Matrix Adaptation Evolution Strategy) adapts the
full covariance matrix of a multivariate Gaussian distribution to
efficiently search continuous spaces. This implementation follows the
(mu/mu_w, lambda)-CMA-ES variant used in Optuna's CmaEsSampler.
"""

_FILE = "scikit-learn/custom_hpo.py"

_CONTENT = """\

class CustomHPOStrategy:
    \"\"\"CMA-ES: Covariance Matrix Adaptation Evolution Strategy.\"\"\"

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self._initialized = False
        self._mean = None
        self._sigma = 0.3
        self._C = None  # covariance matrix
        self._p_sigma = None  # evolution path for sigma
        self._p_c = None  # evolution path for C
        self._gen = 0
        self._lam = None  # population size
        self._mu = None
        self._weights = None
        self._mu_eff = None
        self._candidates = []
        self._pending_evals = []

    def _encode(self, config, space):
        vec = []
        for p in space.params:
            val = config[p.name]
            if p.type == \"categorical\":
                idx = p.choices.index(val)
                vec.append(idx / max(len(p.choices) - 1, 1))
            elif p.type in (\"float\", \"int\"):
                if p.log_scale:
                    v = (np.log(val) - np.log(p.low)) / (np.log(p.high) - np.log(p.low))
                else:
                    v = (val - p.low) / (p.high - p.low)
                vec.append(float(np.clip(v, 0, 1)))
        return np.array(vec)

    def _decode(self, vec, space):
        config = {}
        for i, p in enumerate(space.params):
            v = float(np.clip(vec[i], 0, 1))
            if p.type == \"categorical\":
                idx = int(round(v * max(len(p.choices) - 1, 1)))
                idx = min(idx, len(p.choices) - 1)
                config[p.name] = p.choices[idx]
            elif p.type == \"float\":
                if p.log_scale:
                    config[p.name] = float(np.exp(
                        np.log(p.low) + v * (np.log(p.high) - np.log(p.low))))
                else:
                    config[p.name] = float(p.low + v * (p.high - p.low))
            elif p.type == \"int\":
                if p.log_scale:
                    config[p.name] = int(round(np.exp(
                        np.log(p.low) + v * (np.log(p.high) - np.log(p.low)))))
                else:
                    config[p.name] = int(round(p.low + v * (p.high - p.low)))
        return config

    def _init_cma(self, dim):
        self._mean = np.full(dim, 0.5)
        self._C = np.eye(dim)
        self._p_sigma = np.zeros(dim)
        self._p_c = np.zeros(dim)
        self._lam = 4 + int(3 * np.log(dim))
        self._mu = self._lam // 2
        weights = np.log(self._mu + 0.5) - np.log(np.arange(1, self._mu + 1))
        self._weights = weights / weights.sum()
        self._mu_eff = 1.0 / np.sum(self._weights ** 2)
        self._initialized = True

    def _sample_population(self, space):
        dim = space.dim
        # Eigendecomposition of C
        eigvals, eigvecs = np.linalg.eigh(self._C)
        eigvals = np.maximum(eigvals, 1e-20)
        sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

        self._candidates = []
        self._pending_evals = []
        for _ in range(self._lam):
            z = self.rng.randn(dim)
            x = self._mean + self._sigma * sqrt_C @ z
            x = np.clip(x, 0, 1)
            cfg = self._decode(x, space)
            cfg = space.clip(cfg)
            self._candidates.append((x, cfg, None))
            self._pending_evals.append(cfg)

    def _update(self, space):
        \"\"\"CMA-ES update step after a full generation is evaluated.\"\"\"
        dim = space.dim

        # Sort by score (descending — we maximize)
        scored = [(s, x) for x, _, s in self._candidates if s is not None]
        scored.sort(key=lambda p: p[0], reverse=True)

        # Recombination
        old_mean = self._mean.copy()
        self._mean = np.zeros(dim)
        for i in range(self._mu):
            self._mean += self._weights[i] * scored[i][1]

        # Evolution paths
        c_sigma = (self._mu_eff + 2) / (dim + self._mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((self._mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + self._mu_eff / dim) / (dim + 4 + 2 * self._mu_eff / dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        eigvals, eigvecs = np.linalg.eigh(self._C)
        eigvals = np.maximum(eigvals, 1e-20)
        inv_sqrt_C = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        self._p_sigma = (1 - c_sigma) * self._p_sigma + \\
            np.sqrt(c_sigma * (2 - c_sigma) * self._mu_eff) * \\
            inv_sqrt_C @ (self._mean - old_mean) / self._sigma

        h_sigma = 1.0 if (np.linalg.norm(self._p_sigma) /
                          np.sqrt(1 - (1 - c_sigma) ** (2 * (self._gen + 1)))
                          < (1.4 + 2 / (dim + 1)) * chi_n) else 0.0

        self._p_c = (1 - c_c) * self._p_c + \\
            h_sigma * np.sqrt(c_c * (2 - c_c) * self._mu_eff) * \\
            (self._mean - old_mean) / self._sigma

        # Covariance matrix update
        c1 = 2.0 / ((dim + 1.3) ** 2 + self._mu_eff)
        c_mu = min(1 - c1, 2 * (self._mu_eff - 2 + 1.0 / self._mu_eff) /
                   ((dim + 2) ** 2 + self._mu_eff))

        rank_one = np.outer(self._p_c, self._p_c)
        rank_mu = np.zeros((dim, dim))
        for i in range(self._mu):
            diff = (scored[i][1] - old_mean) / self._sigma
            rank_mu += self._weights[i] * np.outer(diff, diff)

        self._C = (1 - c1 - c_mu) * self._C + c1 * rank_one + c_mu * rank_mu
        # Ensure symmetry and positive definiteness
        self._C = (self._C + self._C.T) / 2
        eigvals_check = np.linalg.eigvalsh(self._C)
        if np.min(eigvals_check) < 1e-20:
            self._C += np.eye(dim) * (1e-20 - np.min(eigvals_check))

        # Step-size update
        self._sigma *= np.exp(
            (c_sigma / d_sigma) * (np.linalg.norm(self._p_sigma) / chi_n - 1))
        self._sigma = np.clip(self._sigma, 1e-10, 1.0)

        self._gen += 1

    def suggest(
        self,
        space: SearchSpace,
        history: List[Trial],
        budget_left: int,
    ) -> Tuple[Dict[str, Any], float]:
        if not self._initialized:
            self._init_cma(space.dim)
            self._sample_population(space)

        # Update scores for pending candidates
        if history:
            last = history[-1]
            last_vec = self._encode(last.config, space)
            for i, (x, cfg, score) in enumerate(self._candidates):
                if score is None and np.allclose(x, last_vec, atol=0.01):
                    self._candidates[i] = (x, cfg, last.score)
                    break

        # If all candidates evaluated, do CMA update and resample
        if self._candidates and all(s is not None for _, _, s in self._candidates):
            self._update(space)
            self._sample_population(space)

        # Return next pending evaluation
        if self._pending_evals:
            cfg = self._pending_evals.pop(0)
            return cfg, 1.0

        # Fallback
        return space.sample_uniform(self.rng), 1.0


"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 255,
        "end_line": 326,
        "content": _CONTENT,
    },
]
