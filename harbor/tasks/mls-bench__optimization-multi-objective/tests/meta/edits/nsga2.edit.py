"""NSGA-II baseline for opt-multi-objective.

Reference: K. Deb, A. Pratap, S. Agarwal, T. Meyarivan (2002).
"A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II."
IEEE Transactions on Evolutionary Computation, 6(2), 182-197.

NSGA-II uses non-dominated sorting and crowding distance for
environmental selection, with binary tournament selection based on
rank and crowding distance for parent selection.
"""

_FILE = "deap/custom_moea.py"

_CONTENT = """\

class CustomMOEA:
    \"\"\"NSGA-II: Non-dominated Sorting Genetic Algorithm II.\"\"\"

    def __init__(self, pop_size, n_obj, n_var, bounds, cx_eta=20.0, mut_eta=20.0, mut_prob=None):
        self.pop_size = pop_size
        self.n_obj = n_obj
        self.n_var = n_var
        self.bounds = bounds
        self.cx_eta = cx_eta
        self.mut_eta = mut_eta
        self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var

    def select(self, population, k):
        \"\"\"Binary tournament selection with crowding distance.\"\"\"
        fronts = tools.sortNondominated(population, len(population), first_front_only=False)
        for front in fronts:
            compute_crowding_distance(front)
        return tools.selTournamentDCD(population, k)

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
        \"\"\"NSGA-II survival: non-dominated sorting + crowding distance.\"\"\"
        combined = population + offspring
        fronts = tools.sortNondominated(combined, self.pop_size, first_front_only=False)

        next_gen = []
        for front in fronts:
            if len(next_gen) + len(front) <= self.pop_size:
                next_gen.extend(front)
            else:
                remaining = self.pop_size - len(next_gen)
                compute_crowding_distance(front)
                front.sort(key=lambda x: x.fitness.crowding_dist, reverse=True)
                next_gen.extend(front[:remaining])
                break

        return next_gen

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
