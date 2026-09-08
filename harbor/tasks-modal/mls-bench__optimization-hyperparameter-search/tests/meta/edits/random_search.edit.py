"""Random Search baseline for opt-hyperparameter-search.

Reference: Bergstra & Bengio (2012). "Random Search for Hyper-Parameter
Optimization." JMLR 13:281-305.

Random search samples configurations uniformly at random from the search
space. Despite its simplicity, it is a strong baseline — especially in
high-dimensional spaces where many dimensions are irrelevant.
"""

_FILE = "scikit-learn/custom_hpo.py"

_CONTENT = """\

class CustomHPOStrategy:
    \"\"\"Random Search: sample configurations uniformly at random.\"\"\"

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def suggest(
        self,
        space: SearchSpace,
        history: List[Trial],
        budget_left: int,
    ) -> Tuple[Dict[str, Any], float]:
        config = space.sample_uniform(self.rng)
        return config, 1.0


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
