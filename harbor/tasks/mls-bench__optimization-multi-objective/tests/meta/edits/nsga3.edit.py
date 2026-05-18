"""NSGA-III baseline for opt-multi-objective.

Reference: K. Deb and H. Jain (2014).
"An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based
Nondominated Sorting Approach, Part I: Solving Problems With Box Constraints."
IEEE Transactions on Evolutionary Computation, 18(4), 577-601.

NSGA-III extends NSGA-II by replacing crowding distance with reference-point-based
niching for better diversity in many-objective spaces.
"""

_FILE = "deap/custom_moea.py"

_CONTENT = """\

class CustomMOEA:
    \"\"\"NSGA-III: Non-dominated Sorting Genetic Algorithm III.\"\"\"

    def __init__(self, pop_size, n_obj, n_var, bounds, cx_eta=20.0, mut_eta=20.0, mut_prob=None):
        self.pop_size = pop_size
        self.n_obj = n_obj
        self.n_var = n_var
        self.bounds = bounds
        self.cx_eta = cx_eta
        self.mut_eta = mut_eta
        self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var

        # Generate reference points
        if n_obj == 2:
            p = pop_size - 1  # number of divisions
            self.ref_points = tools.uniform_reference_points(n_obj, p=p)
        else:
            self.ref_points = tools.uniform_reference_points(n_obj, p=12)

    def select(self, population, k):
        \"\"\"Random shuffle selection (NSGA-III relies on survive for diversity).\"\"\"
        selected = [deepcopy(ind) for ind in population]
        random.shuffle(selected)
        return selected[:k]

    def vary(self, parents):
        \"\"\"SBX crossover + polynomial mutation.\"\"\"
        offspring = [deepcopy(ind) for ind in parents]
        lo, hi = self.bounds

        for i in range(0, len(offspring) - 1, 2):
            if random.random() < 1.0:
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
        \"\"\"NSGA-III survival: reference-point-based selection.\"\"\"
        combined = population + offspring

        # Use DEAP's built-in NSGA-III selection
        selected = tools.selNSGA3(combined, self.pop_size, self.ref_points)
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
