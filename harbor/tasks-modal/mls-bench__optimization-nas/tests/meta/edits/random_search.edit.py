"""Random Search baseline — rigorous codebase edit ops.

Samples random architectures uniformly and tracks the best by val accuracy.
Simplest possible NAS baseline; also the hardest to beat in regimes where
the search space is small and the budget is large.

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "naslib/custom_nas_search.py"

_RANDOM_SEARCH = """\
class NASOptimizer:
    \"\"\"Random Search — uniformly sample architectures and track the best.\"\"\"

    def __init__(self, api, num_epochs, seed):
        self.api = api
        self.num_epochs = num_epochs
        self.seed = seed
        self.best_arch = None
        self.best_val_acc = -1.0

    def search_step(self, epoch):
        arch = random_architecture()
        val_acc = self.api.query_val_accuracy(arch)

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_arch = arch

        return {
            "best_val_acc": self.best_val_acc,
            "queries": self.api.query_count,
            "current_val_acc": val_acc,
        }

    def get_best_architecture(self):
        return self.best_arch
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 163,
        "end_line": 234,
        "content": _RANDOM_SEARCH,
    },
]
