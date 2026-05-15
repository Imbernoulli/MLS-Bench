"""Parsimony-Penalized GP baseline — rigorous codebase edit ops.

Same operators as Standard GP but with a complexity-aware fitness function
that penalizes large expression trees (parsimony pressure). The penalty is
applied only during selection, not in the fitness function itself, so that
the main loop's best-tree tracking uses raw MSE.

ML science contribution: complexity-penalized fitness for bloat control.

Reference: Poli et al. (2008), gplearn parsimony_coefficient=0.001

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "gplearn/custom_sr.py"

_PARSIMONY_GP = """\
def fitness_function(tree, X, y):
    \"\"\"Raw MSE fitness — lower is better.

    Parsimony pressure is applied at the population level inside
    evolve_one_generation, not here. This ensures best_tree_ever
    in the main loop tracks the best-fitting tree by actual MSE.
    \"\"\"
    y_pred = safe_evaluate(tree, X)
    return float(np.mean((y - y_pred) ** 2))


def selection(population, fitnesses, n_select, tournament_size=7):
    \"\"\"Tournament selection on (possibly penalized) fitnesses.\"\"\"
    selected = []
    pop_size = len(population)
    for _ in range(n_select):
        candidates = random.sample(range(pop_size), min(tournament_size, pop_size))
        best = min(candidates, key=lambda i: fitnesses[i])
        selected.append(population[best].copy())
    return selected


def crossover(parent1, parent2, n_features, max_depth=17):
    \"\"\"Standard subtree crossover.\"\"\"
    offspring = parent1.copy()
    donor = parent2.copy()

    off_size = offspring.size()
    don_size = donor.size()
    if off_size <= 1 or don_size <= 1:
        return offspring

    off_point = random.randint(1, off_size - 1)
    don_point = random.randint(0, don_size - 1)

    donor_nodes = donor.get_all_nodes()
    donor_subtree = donor_nodes[don_point][0].copy()

    off_nodes = offspring.get_all_nodes()
    node, parent, child_idx = off_nodes[off_point]
    if parent is not None:
        parent.children[child_idx] = donor_subtree
    else:
        offspring = donor_subtree

    if offspring.depth() > max_depth:
        return parent1.copy()

    return offspring


def mutation(parent, n_features, max_depth=17):
    \"\"\"Subtree mutation — replace a random subtree with a new random tree.\"\"\"
    offspring = parent.copy()
    tree_size = offspring.size()
    if tree_size <= 1:
        return generate_tree('grow', 3, n_features)

    mut_point = random.randint(1, tree_size - 1)
    new_subtree = generate_tree('grow', 3, n_features)

    nodes = offspring.get_all_nodes()
    node, par, child_idx = nodes[mut_point]
    if par is not None:
        par.children[child_idx] = new_subtree
    else:
        offspring = new_subtree

    if offspring.depth() > max_depth:
        return parent.copy()

    return offspring


def evolve_one_generation(population, fitnesses, X_train, y_train,
                          n_features, pop_size,
                          crossover_rate=0.9, mutation_rate=0.05,
                          max_depth=17):
    \"\"\"Parsimony GP generation with parsimony pressure for bloat control.

    Uses gplearn-style auto parsimony coefficient computed per generation:
        c = Cov(length, fitness) / Var(length)
    clamped to [0, 0.001] to prevent runaway penalization.
    Parsimony pressure is applied only during selection; elitism uses
    raw fitness so the best-fitting individual is always preserved.
    \"\"\"
    new_population = []

    # Adaptive parsimony coefficient (gplearn 'auto' method, clamped)
    lengths = np.array([tree.size() for tree in population], dtype=float)
    raw_fit = np.array(fitnesses, dtype=float)
    len_var = float(np.var(lengths))
    if len_var > 1e-15:
        parsimony_coeff = float(np.cov(lengths, raw_fit)[1, 0]) / len_var
        parsimony_coeff = max(parsimony_coeff, 0.0)
        parsimony_coeff = min(parsimony_coeff, 0.001)
    else:
        parsimony_coeff = 0.0

    # Penalized fitnesses for selection only
    penalized = [f + parsimony_coeff * l for f, l in zip(fitnesses, lengths)]

    # Elitism: keep best by raw fitness (not penalized)
    elite_idx = int(np.argmin(fitnesses))
    new_population.append(population[elite_idx].copy())

    while len(new_population) < pop_size:
        r = random.random()
        if r < crossover_rate:
            parents = selection(population, penalized, 2)
            child = crossover(parents[0], parents[1], n_features, max_depth)
        elif r < crossover_rate + mutation_rate:
            parents = selection(population, penalized, 1)
            child = mutation(parents[0], n_features, max_depth)
        else:
            parents = selection(population, penalized, 1)
            child = parents[0]
        new_population.append(child)

    return new_population[:pop_size]
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 228,
        "end_line": 306,
        "content": _PARSIMONY_GP,
    },
]
