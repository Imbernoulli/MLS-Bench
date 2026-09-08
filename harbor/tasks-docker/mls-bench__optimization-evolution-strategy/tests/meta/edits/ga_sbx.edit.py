"""GA baseline: Tournament Selection + SBX Crossover + Polynomial Mutation.

This is the standard genetic algorithm for continuous optimization.
- Selection: Tournament selection (k=3)
- Crossover: Simulated Binary Crossover (SBX), eta=20
- Mutation: Polynomial bounded mutation, eta=20

Reference: Deb & Agrawal (1995), "Simulated Binary Crossover for
Continuous Search Space", Complex Systems 9(2):115-148.
"""

_FILE = "deap/custom_evolution.py"

_CONTENT = """\

def custom_select(population: list, k: int, toolbox=None) -> list:
    \"\"\"Tournament selection with tournament size 3.\"\"\"
    return tools.selTournament(population, k, tournsize=3)


def custom_crossover(ind1: list, ind2: list) -> Tuple[list, list]:
    \"\"\"Simulated Binary Crossover (SBX), eta=20.\"\"\"
    tools.cxSimulatedBinary(ind1, ind2, eta=20.0)
    return ind1, ind2


def custom_mutate(individual: list, lo: float, hi: float) -> Tuple[list]:
    \"\"\"Polynomial bounded mutation, eta=20, indpb=1/dim.\"\"\"
    tools.mutPolynomialBounded(
        individual, eta=20.0, low=lo, up=hi,
        indpb=1.0 / len(individual)
    )
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
    \"\"\"Standard GA with tournament selection, SBX crossover, polynomial mutation.\"\"\"
    random.seed(seed)
    np.random.seed(seed)

    toolbox = base.Toolbox()
    toolbox.register("individual", make_individual, toolbox, dim, lo, hi)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_func)

    pop = toolbox.population(n=pop_size)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    fitness_history = []

    for gen in range(n_generations):
        offspring = custom_select(pop, len(pop), toolbox)
        offspring = [toolbox.clone(ind) for ind in offspring]

        for i in range(0, len(offspring) - 1, 2):
            if random.random() < cx_prob:
                custom_crossover(offspring[i], offspring[i + 1])
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values

        for i in range(len(offspring)):
            if random.random() < mut_prob:
                custom_mutate(offspring[i], lo, hi)
                del offspring[i].fitness.values

        for ind in offspring:
            clip_individual(ind, lo, hi)

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring

        best_fit = min(ind.fitness.values[0] for ind in pop)
        fitness_history.append(best_fit)

        if (gen + 1) % 50 == 0 or gen == 0:
            avg_fit = sum(ind.fitness.values[0] for ind in pop) / len(pop)
            print(
                f"TRAIN_METRICS gen={gen+1} best_fitness={best_fit:.6e} "
                f"avg_fitness={avg_fit:.6e}",
                flush=True,
            )

    best_ind = min(pop, key=lambda ind: ind.fitness.values[0])
    return best_ind, fitness_history
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
