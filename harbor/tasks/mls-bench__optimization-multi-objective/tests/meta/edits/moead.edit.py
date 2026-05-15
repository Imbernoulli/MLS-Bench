"""MOEA/D baseline for opt-multi-objective.

Reference: Q. Zhang and H. Li (2007).
"MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition."
IEEE Transactions on Evolutionary Computation, 11(6), 712-731.

MOEA/D decomposes the multi-objective problem into scalar subproblems
using uniformly distributed weight vectors and Tchebycheff scalarization.
Each subproblem is optimized cooperatively using neighborhood information.
"""

_FILE = "deap/custom_moea.py"

_CONTENT = """\

class CustomMOEA:
    \"\"\"MOEA/D: Multi-Objective Evolutionary Algorithm Based on Decomposition.\"\"\"

    def __init__(self, pop_size, n_obj, n_var, bounds, cx_eta=20.0, mut_eta=20.0, mut_prob=None):
        self.pop_size = pop_size
        self.n_obj = n_obj
        self.n_var = n_var
        self.bounds = bounds
        self.cx_eta = cx_eta
        self.mut_eta = mut_eta
        self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var
        self.T = 20  # neighborhood size
        self.delta = 0.9  # probability of selecting from neighborhood

        # Generate weight vectors
        self.weights = self._generate_weights(pop_size, n_obj)
        self.pop_size = len(self.weights)  # adjust to actual number of weight vectors

        # Compute neighborhoods
        self.neighbors = self._compute_neighborhoods()

        # Ideal point (updated during search)
        self.z_star = None

    def _generate_weights(self, n, n_obj):
        \"\"\"Generate uniformly distributed weight vectors.\"\"\"
        if n_obj == 2:
            weights = []
            for i in range(n):
                w1 = i / max(n - 1, 1)
                weights.append([w1, 1.0 - w1])
            return np.array(weights)
        else:
            # Use DEAP's uniform reference points for 3+ objectives
            ref_points = tools.uniform_reference_points(n_obj, p=12)
            return np.array(ref_points)

    def _compute_neighborhoods(self):
        \"\"\"Compute T-nearest weight vector neighborhoods.\"\"\"
        from scipy.spatial.distance import cdist
        dist_matrix = cdist(self.weights, self.weights)
        neighbors = []
        for i in range(len(self.weights)):
            idx = np.argsort(dist_matrix[i])[:self.T]
            neighbors.append(idx.tolist())
        return neighbors

    def _tchebycheff(self, fitness_values, weight, z_star):
        \"\"\"Tchebycheff scalarization.\"\"\"
        return max(weight[j] * abs(fitness_values[j] - z_star[j])
                   for j in range(self.n_obj))

    def select(self, population, k):
        \"\"\"MOEA/D doesn't use standard selection — return population as-is.\"\"\"
        return [deepcopy(ind) for ind in population]

    def vary(self, parents):
        \"\"\"Generate one offspring per subproblem using neighborhood mating.\"\"\"
        offspring = []
        lo, hi = self.bounds

        for i in range(len(parents)):
            # Select mating pool (neighborhood or whole population)
            if random.random() < self.delta:
                pool = [parents[j] for j in self.neighbors[i % len(self.neighbors)]]
            else:
                pool = parents

            # Select two parents from pool
            p1, p2 = random.sample(range(len(pool)), 2)
            child = deepcopy(pool[p1])

            # SBX crossover
            mate = deepcopy(pool[p2])
            if random.random() < 1.0:
                tools.cxSimulatedBinaryBounded(child, mate, eta=self.cx_eta, low=lo, up=hi)

            # Polynomial mutation
            tools.mutPolynomialBounded(child, eta=self.mut_eta, low=lo, up=hi, indpb=self.mut_prob)
            del child.fitness.values
            offspring.append(child)

        return offspring

    def survive(self, population, offspring):
        \"\"\"MOEA/D survival: update subproblems using Tchebycheff decomposition.\"\"\"
        # Update ideal point
        all_inds = [ind for ind in population + offspring if ind.fitness.valid]
        if not all_inds:
            return population

        if self.z_star is None:
            self.z_star = [float('inf')] * self.n_obj
        for ind in all_inds:
            for j in range(self.n_obj):
                if ind.fitness.values[j] < self.z_star[j]:
                    self.z_star[j] = ind.fitness.values[j]

        # Update each subproblem
        next_gen = list(population)
        for i in range(min(len(offspring), len(self.weights))):
            child = offspring[i]
            if not child.fitness.valid:
                continue

            # Update neighbors
            neighbors_idx = self.neighbors[i % len(self.neighbors)]
            for j_idx in neighbors_idx:
                if j_idx >= len(next_gen):
                    continue
                g_child = self._tchebycheff(child.fitness.values, self.weights[j_idx], self.z_star)
                g_current = self._tchebycheff(next_gen[j_idx].fitness.values, self.weights[j_idx], self.z_star)
                if g_child < g_current:
                    next_gen[j_idx] = deepcopy(child)

        return next_gen[:self.pop_size]

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
