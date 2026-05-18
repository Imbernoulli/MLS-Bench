"""SPEA2 baseline for opt-multi-objective.

Reference: E. Zitzler, M. Laumanns, L. Thiele (2001).
"SPEA2: Improving the Strength Pareto Evolutionary Algorithm."
TIK Report 103, ETH Zurich.

SPEA2 uses strength-based fitness assignment with k-nearest-neighbor
density estimation. An external archive maintains elite solutions.
"""

_FILE = "deap/custom_moea.py"

_CONTENT = """\

class CustomMOEA:
    \"\"\"SPEA2: Strength Pareto Evolutionary Algorithm 2.\"\"\"

    def __init__(self, pop_size, n_obj, n_var, bounds, cx_eta=20.0, mut_eta=20.0, mut_prob=None):
        self.pop_size = pop_size
        self.n_obj = n_obj
        self.n_var = n_var
        self.bounds = bounds
        self.cx_eta = cx_eta
        self.mut_eta = mut_eta
        self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var
        self.archive = []

    def select(self, population, k):
        \"\"\"Binary tournament selection using SPEA2 fitness from archive.\"\"\"
        # Use archive for selection if available, otherwise population
        pool = self.archive if self.archive else population
        # Binary tournament on dominance
        selected = []
        for _ in range(k):
            i1, i2 = random.sample(range(len(pool)), 2)
            a, b = pool[i1], pool[i2]
            if a.fitness.dominates(b.fitness):
                selected.append(deepcopy(a))
            elif b.fitness.dominates(a.fitness):
                selected.append(deepcopy(b))
            else:
                selected.append(deepcopy(random.choice([a, b])))
        return selected

    def vary(self, parents):
        \"\"\"SBX crossover + polynomial mutation.\"\"\"
        offspring = [deepcopy(ind) for ind in parents]
        lo, hi = self.bounds

        for i in range(0, len(offspring) - 1, 2):
            if random.random() < 0.9:
                tools.cxSimulatedBinaryBounded(
                    offspring[i], offspring[i + 1],
                    eta=self.cx_eta, low=lo, up=hi,
                )
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values

        for ind in offspring:
            if random.random() < 1.0:
                tools.mutPolynomialBounded(
                    ind, eta=self.mut_eta, low=lo, up=hi, indpb=self.mut_prob,
                )
                del ind.fitness.values

        return offspring

    def survive(self, population, offspring):
        \"\"\"SPEA2 survival: strength fitness + kNN density truncation.\"\"\"
        combined = population + offspring

        # Use DEAP's built-in SPEA2 selection
        selected = tools.selSPEA2(combined, self.pop_size)

        # Update archive with non-dominated solutions
        nd = get_nondominated(selected)
        self.archive = [deepcopy(ind) for ind in nd[:self.pop_size]]

        return selected

    def on_generation(self, gen, population):
        pass


"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 297,
        "end_line": 441,
        "content": _CONTENT,
    },
]
