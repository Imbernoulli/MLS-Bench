"""Differential Evolution (DE/rand/1/bin) baseline.

Classic DE with rand/1 mutation and binomial crossover.
- Mutation: v = x_r1 + F * (x_r2 - x_r3), F=0.5
- Crossover: binomial with CR=0.9
- Selection: greedy (offspring replaces parent if better)

Reference: Storn & Price (1997), "Differential Evolution — A Simple
and Efficient Heuristic for Global Optimization over Continuous Spaces",
Journal of Global Optimization 11(4):341-359.
"""

_FILE = "deap/custom_evolution.py"

_CONTENT = """\

def custom_select(population: list, k: int, toolbox=None) -> list:
    \"\"\"Not used directly in DE (greedy selection is built into run_evolution).\"\"\"
    return population[:k]


def custom_crossover(ind1: list, ind2: list) -> Tuple[list, list]:
    \"\"\"Not used directly in DE (binomial crossover is built into run_evolution).\"\"\"
    return ind1, ind2


def custom_mutate(individual: list, lo: float, hi: float) -> Tuple[list]:
    \"\"\"Not used directly in DE (DE mutation is built into run_evolution).\"\"\"
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
    \"\"\"Differential Evolution: DE/rand/1/bin.

    F (scale factor) = 0.5, CR (crossover rate) = 0.9.
    Greedy selection: trial replaces target only if fitness improves.
    \"\"\"
    random.seed(seed)
    np.random.seed(seed)

    F = 0.5   # Scale factor (Storn & Price recommended default)
    CR = 0.9  # Crossover rate

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
        for i in range(len(pop)):
            # Select three distinct random individuals (not i)
            candidates = list(range(len(pop)))
            candidates.remove(i)
            r1, r2, r3 = random.sample(candidates, 3)
            x_r1, x_r2, x_r3 = pop[r1], pop[r2], pop[r3]

            # DE/rand/1 mutation
            mutant = creator.Individual([
                x_r1[j] + F * (x_r2[j] - x_r3[j])
                for j in range(dim)
            ])

            # Binomial crossover
            j_rand = random.randint(0, dim - 1)
            trial = creator.Individual([
                mutant[j] if (random.random() < CR or j == j_rand) else pop[i][j]
                for j in range(dim)
            ])

            # Clip to bounds
            for j in range(dim):
                trial[j] = max(lo, min(hi, trial[j]))

            # Evaluate trial
            trial.fitness.values = toolbox.evaluate(trial)

            # Greedy selection
            if trial.fitness.values[0] <= pop[i].fitness.values[0]:
                pop[i] = trial

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
