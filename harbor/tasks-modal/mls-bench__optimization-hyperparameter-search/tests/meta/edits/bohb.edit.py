"""BOHB (Bayesian Optimization and Hyperband) baseline for opt-hyperparameter-search.

Reference: Falkner, Klein & Hutter (2018). "BOHB: Robust and Efficient
Hyperparameter Optimization at Scale." ICML.

BOHB replaces the random configuration sampling in Hyperband with a
TPE-based model. It builds kernel density estimators on the observed
configurations, split by performance, and samples from the better
distribution — combining the principled resource allocation of Hyperband
with the model-guided search of TPE.
"""

_FILE = "scikit-learn/custom_hpo.py"

_CONTENT = """\

class CustomHPOStrategy:
    \"\"\"BOHB: Bayesian Optimization + Hyperband.\"\"\"

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.eta = 3
        self.gamma = 0.15  # fraction for good KDE
        self.n_startup = 8  # random configs before model-guided
        self.n_candidates = 24
        self.bw_factor = 1.0
        self._brackets = []
        self._queue = []
        self._initialized = False
        self._all_trials = []  # (vec, score, fidelity)

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

    def _kde_logpdf(self, x, samples, bw):
        diffs = x[None, :] - samples
        return float(np.log(
            np.mean(np.exp(-0.5 * np.sum(diffs**2 / bw**2, axis=1))) + 1e-30
        ))

    def _sample_from_model(self, space):
        \"\"\"Sample config guided by TPE model or random.\"\"\"
        if len(self._all_trials) < self.n_startup:
            return space.sample_uniform(self.rng)

        vecs = np.array([t[0] for t in self._all_trials])
        scores = np.array([t[1] for t in self._all_trials])
        n_good = max(1, int(self.gamma * len(scores)))
        threshold = np.sort(scores)[-n_good]

        good = vecs[scores >= threshold]
        bad = vecs[scores < threshold]
        if len(bad) == 0:
            bad = good.copy()

        bw_good = max(0.05, good.std() * self.bw_factor + 1e-6)
        bw_bad = max(0.05, bad.std() * self.bw_factor + 1e-6)

        best_ei = -np.inf
        best_cfg = None
        for _ in range(self.n_candidates):
            cfg = space.sample_uniform(self.rng)
            x = self._encode(cfg, space)
            log_l = self._kde_logpdf(x, good, bw_good)
            log_g = self._kde_logpdf(x, bad, bw_bad)
            ei = log_l - log_g
            if ei > best_ei:
                best_ei = ei
                best_cfg = cfg
        return best_cfg

    def _init_brackets(self, space, total_budget):
        s_max = max(0, int(np.floor(np.log(total_budget) / np.log(self.eta))))
        s_max = min(s_max, 3)

        for s in range(s_max, -1, -1):
            n = int(np.ceil((s_max + 1) / (s + 1)) * self.eta ** s)
            n = min(n, total_budget)
            r = max(1.0 / self.eta ** s, 0.1)

            configs = [self._sample_from_model(space) for _ in range(n)]
            for cfg in configs:
                self._queue.append((cfg, r))

            self._brackets.append({
                \"configs\": configs,
                \"fidelity\": r,
                \"scores\": [None] * len(configs),
            })

    def suggest(
        self,
        space: SearchSpace,
        history: List[Trial],
        budget_left: int,
    ) -> Tuple[Dict[str, Any], float]:
        # Track all completed trials for the TPE model
        if history:
            last = history[-1]
            vec = self._encode(last.config, space)
            self._all_trials.append((vec, last.score, last.budget))

        if not self._initialized:
            self._init_brackets(space, budget_left + len(history))
            self._initialized = True

        # Update bracket scores
        if history:
            last = history[-1]
            for bracket in self._brackets:
                for i, cfg in enumerate(bracket[\"configs\"]):
                    if (bracket[\"scores\"][i] is None
                            and cfg == last.config
                            and abs(bracket[\"fidelity\"] - last.budget) < 0.05):
                        bracket[\"scores\"][i] = last.score

                # Advance complete brackets
                if all(s is not None for s in bracket[\"scores\"]):
                    if bracket[\"fidelity\"] < 1.0 and len(bracket[\"configs\"]) > 1:
                        # Successive halving
                        paired = list(zip(bracket[\"scores\"], bracket[\"configs\"]))
                        paired.sort(key=lambda x: x[0], reverse=True)
                        n_keep = max(1, len(paired) // self.eta)
                        survivors = paired[:n_keep]
                        new_fid = min(bracket[\"fidelity\"] * self.eta, 1.0)
                        bracket[\"configs\"] = [c for _, c in survivors]
                        bracket[\"scores\"] = [None] * len(survivors)
                        bracket[\"fidelity\"] = new_fid
                        for cfg in bracket[\"configs\"]:
                            self._queue.append((cfg, new_fid))

        if self._queue:
            return self._queue.pop(0)

        # Generate new configs using TPE model at full fidelity
        cfg = self._sample_from_model(space)
        return cfg, 1.0


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
