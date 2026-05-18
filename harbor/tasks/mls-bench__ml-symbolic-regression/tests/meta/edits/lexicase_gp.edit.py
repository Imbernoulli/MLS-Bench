"""Lexicase GP baseline — rigorous codebase edit ops.

Standard GP operators but with epsilon-lexicase selection instead of
tournament selection. Lexicase selects parents based on performance
on individual training cases in random order.

ML science contribution: case-based selection for improved generalization.

Reference: Spector et al. (2012), La Cava et al. (2016) — epsilon-lexicase

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "gplearn/custom_sr.py"

_LEXICASE_GP = """\
def fitness_function(tree, X, y):
    \"\"\"MSE fitness — lower is better.\"\"\"
    y_pred = safe_evaluate(tree, X)
    return float(np.mean((y - y_pred) ** 2))


def _per_case_errors(population, X, y):
    \"\"\"Compute per-case absolute errors for the entire population.

    Returns:
        numpy array of shape (len(population), n_samples)
    \"\"\"
    errors = np.empty((len(population), X.shape[0]))
    for i, tree in enumerate(population):
        y_pred = safe_evaluate(tree, X)
        errors[i] = np.abs(y - y_pred)
    return errors


def selection(population, fitnesses, n_select, _errors=None, _X=None, _y=None):
    \"\"\"Epsilon-lexicase selection.

    Requires _errors (per-case errors), _X, _y to be passed via
    evolve_one_generation. Falls back to tournament if not available.
    \"\"\"
    selected = []
    pop_size = len(population)

    if _errors is None:
        # Fallback to tournament
        for _ in range(n_select):
            candidates = random.sample(range(pop_size), min(7, pop_size))
            best = min(candidates, key=lambda i: fitnesses[i])
            selected.append(population[best].copy())
        return selected

    n_cases = _errors.shape[1]
    for _ in range(n_select):
        candidates = list(range(pop_size))
        cases = list(range(n_cases))
        random.shuffle(cases)

        for case in cases:
            if len(candidates) <= 1:
                break
            case_errors = _errors[candidates, case]
            # Semi-dynamic epsilon-lexicase (La Cava 2016/2019): candidates
            # survive iff their error ≤ best_on_case + MAD. The previous
            # `median + MAD` admitted most of the population and degraded
            # lexicase toward random selection.
            min_err = float(np.min(case_errors))
            mad = float(np.median(np.abs(case_errors - float(np.median(case_errors)))))
            candidates = [c for c, e in zip(candidates, case_errors) if e <= min_err + mad]

        winner = random.choice(candidates)
        selected.append(population[winner].copy())

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
    \"\"\"Lexicase GP generation — uses epsilon-lexicase selection.\"\"\"
    new_population = []

    # Elitism: keep best
    elite_idx = int(np.argmin(fitnesses))
    new_population.append(population[elite_idx].copy())

    # Pre-compute per-case errors for lexicase selection
    errors = _per_case_errors(population, X_train, y_train)

    while len(new_population) < pop_size:
        r = random.random()
        if r < crossover_rate:
            parents = selection(population, fitnesses, 2, _errors=errors)
            child = crossover(parents[0], parents[1], n_features, max_depth)
        elif r < crossover_rate + mutation_rate:
            parents = selection(population, fitnesses, 1, _errors=errors)
            child = mutation(parents[0], n_features, max_depth)
        else:
            parents = selection(population, fitnesses, 1, _errors=errors)
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
        "content": _LEXICASE_GP,
    },
]
