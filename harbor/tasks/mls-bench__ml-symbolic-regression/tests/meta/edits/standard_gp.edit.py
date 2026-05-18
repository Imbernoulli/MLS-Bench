"""Standard GP baseline — rigorous codebase edit ops.

Tournament selection, subtree crossover, subtree mutation, raw MSE fitness.
The classic Koza-style genetic programming approach.

Reference: gplearn/gplearn/genetic.py, gplearn/gplearn/_program.py

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "gplearn/custom_sr.py"

_STANDARD_GP = """\
def fitness_function(tree, X, y):
    \"\"\"MSE fitness — lower is better.\"\"\"
    y_pred = safe_evaluate(tree, X)
    return float(np.mean((y - y_pred) ** 2))


def selection(population, fitnesses, n_select, tournament_size=7):
    \"\"\"Tournament selection.\"\"\"
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

    # Pick random crossover points
    off_size = offspring.size()
    don_size = donor.size()
    if off_size <= 1 or don_size <= 1:
        return offspring

    off_point = random.randint(1, off_size - 1)
    don_point = random.randint(0, don_size - 1)

    # Extract donor subtree
    donor_nodes = donor.get_all_nodes()
    donor_subtree = donor_nodes[don_point][0].copy()

    # Replace in offspring
    off_nodes = offspring.get_all_nodes()
    node, parent, child_idx = off_nodes[off_point]
    if parent is not None:
        parent.children[child_idx] = donor_subtree
    else:
        offspring = donor_subtree

    # Reject if too deep
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
    \"\"\"Standard GP generation with elitism.\"\"\"
    new_population = []

    # Elitism: keep best
    elite_idx = int(np.argmin(fitnesses))
    new_population.append(population[elite_idx].copy())

    while len(new_population) < pop_size:
        r = random.random()
        if r < crossover_rate:
            parents = selection(population, fitnesses, 2)
            child = crossover(parents[0], parents[1], n_features, max_depth)
        elif r < crossover_rate + mutation_rate:
            parents = selection(population, fitnesses, 1)
            child = mutation(parents[0], n_features, max_depth)
        else:
            parents = selection(population, fitnesses, 1)
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
        "content": _STANDARD_GP,
    },
]
