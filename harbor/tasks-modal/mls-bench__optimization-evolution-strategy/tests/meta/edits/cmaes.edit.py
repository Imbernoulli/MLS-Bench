"""CMA-ES baseline: Covariance Matrix Adaptation Evolution Strategy.

CMA-ES is a state-of-the-art derivative-free optimizer for continuous
domains. It adapts the full covariance matrix of a multivariate normal
distribution to guide the search.

Reference: Hansen & Ostermeier (2001), "Completely Derandomized
Self-Adaptation in Evolution Strategies", Evolutionary Computation 9(2):159-195.

Uses DEAP's built-in cma.Strategy implementation.
"""

_FILE = "deap/custom_evolution.py"

_CONTENT = """\

def custom_select(population: list, k: int, toolbox=None) -> list:
    \"\"\"Not used in CMA-ES (strategy handles selection internally).\"\"\"
    return tools.selBest(population, k)


def custom_crossover(ind1: list, ind2: list) -> Tuple[list, list]:
    \"\"\"Not used in CMA-ES.\"\"\"
    return ind1, ind2


def custom_mutate(individual: list, lo: float, hi: float) -> Tuple[list]:
    \"\"\"Not used in CMA-ES.\"\"\"
    return (individual,)


def run_evolution(
    evaluate_func: Callable,
    dim: int,
    lo: float,
    hi: float,
    pop_size: int,
    n_generations: int,
    cx_prob: float,
    mut_prob: float,
    seed: int,
) -> Tuple[list, list]:
    \"\"\"CMA-ES: Covariance Matrix Adaptation Evolution Strategy.

    Uses DEAP's cma.Strategy. The population size is set by CMA-ES's
    internal heuristic (lambda ~ 4 + 3*ln(dim)), but we use the provided
    pop_size. Initial sigma is set to 1/3 of the domain range.
    \"\"\"
    from deap import cma as deap_cma

    random.seed(seed)
    np.random.seed(seed)

    # Initial centroid: center of the domain
    centroid = [(lo + hi) / 2.0] * dim
    # Initial step size: ~1/3 of domain width
    sigma = (hi - lo) / 3.0

    strategy = deap_cma.Strategy(
        centroid=centroid,
        sigma=sigma,
        lambda_=pop_size,
    )

    toolbox = base.Toolbox()
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)
    toolbox.register("evaluate", evaluate_func)

    fitness_history = []
    best_ever = None
    best_ever_fit = float("inf")

    for gen in range(n_generations):
        # Generate new population from the CMA-ES distribution
        population = toolbox.generate()

        # Clip to bounds
        for ind in population:
            for i in range(len(ind)):
                ind[i] = max(lo, min(hi, ind[i]))

        # Evaluate
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Update the strategy (covariance matrix, step size, mean)
        toolbox.update(population)

        # Track best
        gen_best = min(population, key=lambda ind: ind.fitness.values[0])
        gen_best_fit = gen_best.fitness.values[0]
        if gen_best_fit < best_ever_fit:
            best_ever_fit = gen_best_fit
            best_ever = creator.Individual(gen_best[:])
            best_ever.fitness.values = gen_best.fitness.values

        fitness_history.append(best_ever_fit)

        if (gen + 1) % 50 == 0 or gen == 0:
            avg_fit = sum(ind.fitness.values[0] for ind in population) / len(population)
            print(
                f"TRAIN_METRICS gen={gen+1} best_fitness={best_ever_fit:.6e} "
                f"avg_fitness={avg_fit:.6e}",
                flush=True,
            )

    return best_ever, fitness_history
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 87,
        "end_line": 225,
        "content": _CONTENT,
    },
]
