"""Hyperband baseline for opt-hyperparameter-search.

Reference: Li, Jamieson, DeSalvo, Rostamizadeh & Talwalkar (2017).
"Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization."
JMLR 18(185):1-52.

Hyperband extends Successive Halving with an outer loop over different
tradeoffs between number of configurations and budget per configuration.
It adaptively allocates resources to promising configurations.
"""

_FILE = "scikit-learn/custom_hpo.py"

_CONTENT = """\

class CustomHPOStrategy:
    \"\"\"Hyperband: multi-fidelity with successive halving.\"\"\"

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.eta = 3  # halving rate
        self.brackets = []  # list of (configs, fidelity, scores)
        self._initialized = False
        self._queue = []  # queue of (config, fidelity) to suggest

    def _init_brackets(self, space, total_budget):
        \"\"\"Initialize Hyperband brackets (Successive Halving instances).\"\"\"
        s_max = max(0, int(np.floor(np.log(total_budget) / np.log(self.eta))))
        s_max = min(s_max, 4)  # cap brackets

        for s in range(s_max, -1, -1):
            n = int(np.ceil((s_max + 1) / (s + 1)) * self.eta ** s)
            n = min(n, total_budget)
            r = max(1.0 / self.eta ** s, 0.1)

            # Generate random configs for this bracket
            configs = [space.sample_uniform(self.rng) for _ in range(n)]
            # Queue low-fidelity evaluations
            for cfg in configs:
                self._queue.append((cfg, r))

            self.brackets.append({
                \"configs\": configs,
                \"fidelity\": r,
                \"scores\": [None] * len(configs),
                \"round\": 0,
                \"s\": s,
            })

    def _advance_bracket(self, bracket):
        \"\"\"Advance a bracket: keep top 1/eta, increase fidelity.\"\"\"
        configs = bracket[\"configs\"]
        scores = bracket[\"scores\"]

        # Sort by score, keep top 1/eta
        paired = [(s, c) for s, c in zip(scores, configs) if s is not None]
        if not paired:
            return
        paired.sort(key=lambda x: x[0], reverse=True)
        n_keep = max(1, len(paired) // self.eta)
        survivors = paired[:n_keep]

        new_fidelity = min(bracket[\"fidelity\"] * self.eta, 1.0)
        bracket[\"configs\"] = [c for _, c in survivors]
        bracket[\"scores\"] = [None] * len(survivors)
        bracket[\"fidelity\"] = new_fidelity
        bracket[\"round\"] += 1

        for cfg in bracket[\"configs\"]:
            self._queue.append((cfg, new_fidelity))

    def suggest(
        self,
        space: SearchSpace,
        history: List[Trial],
        budget_left: int,
    ) -> Tuple[Dict[str, Any], float]:
        if not self._initialized:
            self._init_brackets(space, budget_left + len(history))
            self._initialized = True

        # Update bracket scores from history
        if history:
            last = history[-1]
            for bracket in self.brackets:
                for i, cfg in enumerate(bracket[\"configs\"]):
                    if (bracket[\"scores\"][i] is None
                            and cfg == last.config
                            and abs(bracket[\"fidelity\"] - last.budget) < 0.05):
                        bracket[\"scores\"][i] = last.score

                # Check if bracket round complete
                if all(s is not None for s in bracket[\"scores\"]):
                    if bracket[\"fidelity\"] < 1.0 and len(bracket[\"configs\"]) > 1:
                        self._advance_bracket(bracket)

        # Return from queue
        if self._queue:
            return self._queue.pop(0)

        # Fallback: random
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
