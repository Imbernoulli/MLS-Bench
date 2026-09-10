"""REA (Regularized Evolution Algorithm) baseline — rigorous codebase edit ops.

Maintains a small population of architectures (population_size=10 for the
K=30 sample-efficient regime), evolves by tournament selection and
single-edge mutation, with oldest members removed (regularization).

Reference: Real et al., 2019: Regularized Evolution for Image Classifier
Architecture Search.

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "naslib/custom_nas_search.py"

_REA = """\
class NASOptimizer:
    \"\"\"REA — Regularized Evolution Algorithm for NAS (low-budget variant).

    Population size 10 and tournament size 3, tuned for K=30 queries
    following NAS-Bench-Suite (White et al., 2022) low-budget recipes.
    \"\"\"

    def __init__(self, api, num_epochs, seed):
        self.api = api
        self.num_epochs = num_epochs
        self.seed = seed

        self.population_size = 10
        self.tournament_size = 3
        self.population = []  # list of (arch, val_acc)
        self.best_arch = None
        self.best_val_acc = -1.0

    def _update_best(self, arch, val_acc):
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_arch = list(arch)

    def search_step(self, epoch):
        if epoch < self.population_size:
            # Seed initial population with random architectures
            arch = random_architecture()
            val_acc = self.api.query_val_accuracy(arch)
            self.population.append((arch, val_acc))
        else:
            # Tournament selection
            k = min(self.tournament_size, len(self.population))
            sample_indices = random.sample(range(len(self.population)), k)
            parent_idx = max(sample_indices, key=lambda i: self.population[i][1])
            parent_arch = self.population[parent_idx][0]

            # Mutation
            child_arch = mutate_architecture(parent_arch)
            while not is_valid_arch(child_arch):
                child_arch = mutate_architecture(parent_arch)
            child_val_acc = self.api.query_val_accuracy(child_arch)

            # Add child and remove oldest (regularization)
            self.population.append((child_arch, child_val_acc))
            self.population.pop(0)
            arch, val_acc = child_arch, child_val_acc

        self._update_best(arch, val_acc)

        return {
            "best_val_acc": self.best_val_acc,
            "queries": self.api.query_count,
            "population_size": len(self.population),
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
        "content": _REA,
    },
]
