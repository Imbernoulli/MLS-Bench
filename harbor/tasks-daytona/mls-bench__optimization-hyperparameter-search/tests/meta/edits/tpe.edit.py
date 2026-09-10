"""Tree-structured Parzen Estimator (TPE) baseline for opt-hyperparameter-search.

Reference: Bergstra, Bardenet, Bengio & Kegl (2011). "Algorithms for
Hyper-Parameter Optimization." NeurIPS.

TPE models the conditional densities p(x|y<y*) and p(x|y>=y*) using
kernel density estimators, then suggests configurations that maximize
the ratio l(x)/g(x) — equivalent to maximizing Expected Improvement.
"""

_FILE = "scikit-learn/custom_hpo.py"

_CONTENT = """\

class CustomHPOStrategy:
    \"\"\"Tree-structured Parzen Estimator (TPE).\"\"\"

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.gamma = 0.25  # fraction of best observations for l(x)
        self.n_startup = 10  # random search before modelling
        self.n_ei_candidates = 24  # candidates to score with EI

    def _encode(self, config, space):
        \"\"\"Encode a config to a numeric vector in [0,1].\"\"\"
        vec = []
        for p in space.params:
            val = config[p.name]
            if p.type == \"categorical\":
                # One-hot-ish: use index / len
                idx = p.choices.index(val)
                vec.append(idx / max(len(p.choices) - 1, 1))
            elif p.type in (\"float\", \"int\"):
                if p.log_scale:
                    v = (np.log(val) - np.log(p.low)) / (np.log(p.high) - np.log(p.low))
                else:
                    v = (val - p.low) / (p.high - p.low)
                vec.append(float(np.clip(v, 0, 1)))
        return np.array(vec)

    def _kde_logpdf(self, x, samples, bw):
        \"\"\"Simple Gaussian KDE log-density at x.\"\"\"
        diffs = x[None, :] - samples  # (n_samples, dim)
        return float(
            np.log(np.mean(np.exp(-0.5 * np.sum(diffs**2 / bw**2, axis=1))) + 1e-30)
        )

    def suggest(
        self,
        space: SearchSpace,
        history: List[Trial],
        budget_left: int,
    ) -> Tuple[Dict[str, Any], float]:
        if len(history) < self.n_startup:
            return space.sample_uniform(self.rng), 1.0

        # Split observations into good (l) and bad (g)
        scores = np.array([t.score for t in history])
        n_good = max(1, int(self.gamma * len(history)))
        threshold = np.sort(scores)[-n_good]

        good_vecs = np.array([
            self._encode(t.config, space)
            for t in history if t.score >= threshold
        ])
        bad_vecs = np.array([
            self._encode(t.config, space)
            for t in history if t.score < threshold
        ])

        if len(bad_vecs) == 0:
            bad_vecs = good_vecs.copy()

        # Bandwidth: Scott's rule
        bw_good = max(0.05, good_vecs.std() + 1e-6)
        bw_bad = max(0.05, bad_vecs.std() + 1e-6)

        # Generate candidates and score them by l(x)/g(x)
        best_score = -np.inf
        best_config = None
        for _ in range(self.n_ei_candidates):
            candidate = space.sample_uniform(self.rng)
            x = self._encode(candidate, space)
            log_l = self._kde_logpdf(x, good_vecs, bw_good)
            log_g = self._kde_logpdf(x, bad_vecs, bw_bad)
            ei = log_l - log_g
            if ei > best_score:
                best_score = ei
                best_config = candidate

        return best_config, 1.0


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
