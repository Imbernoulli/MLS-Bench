"""L-SHADE baseline: Success-History based Adaptive DE with Linear
Population Size Reduction.

Adaptive F and CR via success history (Cauchy for F, Normal for CR),
current-to-pbest/1 mutation, and linear population size reduction.

Reference: Tanabe & Fukunaga (2014), "Improving the Search Performance
of SHADE Using Linear Population Size Reduction", IEEE CEC 2014.
"""

_FILE = "deap/custom_evolution.py"

_CONTENT = """\

def custom_select(population: list, k: int, toolbox=None) -> list:
    \"\"\"Not used in L-SHADE (adaptive DE handles selection internally).\"\"\"
    return population[:k]


def custom_crossover(ind1: list, ind2: list) -> Tuple[list, list]:
    \"\"\"Not used in L-SHADE (binomial crossover built into run_evolution).\"\"\"
    return ind1, ind2


def custom_mutate(individual: list, lo: float, hi: float) -> Tuple[list]:
    \"\"\"Not used in L-SHADE (adaptive mutation built into run_evolution).\"\"\"
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
    \"\"\"L-SHADE: Success-History based Adaptive DE with Linear Population Reduction.

    - Adaptive F (Cauchy) and CR (Normal) from success history
    - current-to-pbest/1 mutation with external archive
    - Linear population size reduction from N_init to N_min
    \"\"\"
    random.seed(seed)
    np.random.seed(seed)

    # --- Hyperparameters (Tanabe & Fukunaga, CEC 2014) ---
    # The paper recommends N_init = 18·D, but on small fixed budgets (as in
    # our 400 pop × 1000 gen setting) that value starves the search of
    # generations: on Rastrigin-100D, N_init=1800 with matched total-eval
    # budget degraded from 128 → 313. Use pop_size as given and the
    # canonical N_min = 4 (paper §III-B), which lets the linear population
    # reduction actually run. Budget stays identical to CMA-ES/DE/GA.
    H = 6  # History size (paper default)
    N_init = pop_size
    N_min = 4  # Minimum population size
    p_min = 2.0 / N_init  # Minimum p for pbest
    p_max = 0.2  # Maximum p for pbest

    toolbox = base.Toolbox()
    toolbox.register("individual", make_individual, toolbox, dim, lo, hi)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_func)

    # Initialize population
    pop = toolbox.population(n=N_init)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Success history for F and CR
    M_F = [0.5] * H
    M_CR = [0.5] * H
    k = 0  # History index

    # External archive of inferior solutions
    archive = []

    fitness_history = []

    for gen in range(n_generations):
        N_current = len(pop)

        # Collect successful F and CR values and their fitness improvements
        S_F = []
        S_CR = []
        delta_f = []  # fitness improvement for weighting

        trial_list = []
        F_list = []
        CR_list = []

        for i in range(N_current):
            # Sample F from Cauchy(M_F[r], 0.1), truncate to (0, 1]
            r = random.randint(0, H - 1)
            while True:
                F_i = M_F[r] + 0.1 * np.random.standard_cauchy()
                if F_i > 0:
                    break
            F_i = min(F_i, 1.0)

            # Sample CR from Normal(M_CR[r], 0.1), clamp to [0, 1]
            CR_i = np.random.normal(M_CR[r], 0.1)
            CR_i = max(0.0, min(1.0, CR_i))

            F_list.append(F_i)
            CR_list.append(CR_i)

            # current-to-pbest/1 mutation
            # Choose p_i uniformly from [p_min, p_max]
            p_i = random.uniform(p_min, p_max)
            n_pbest = max(1, int(round(p_i * N_current)))
            sorted_pop = sorted(pop, key=lambda ind: ind.fitness.values[0])
            pbest = random.choice(sorted_pop[:n_pbest])

            # Select r1 from pop (r1 != i)
            candidates = list(range(N_current))
            candidates.remove(i)
            r1 = random.choice(candidates)

            # Select r2 from pop + archive (r2 != i and r2 != r1)
            union = list(range(N_current + len(archive)))
            union_exclude = {i, r1}
            union_avail = [x for x in union if x not in union_exclude]
            if not union_avail:
                union_avail = [x for x in union if x != i]
            r2_idx = random.choice(union_avail)
            if r2_idx < N_current:
                x_r2 = pop[r2_idx]
            else:
                x_r2 = archive[r2_idx - N_current]

            # Mutation: v = x_i + F * (pbest - x_i) + F * (x_r1 - x_r2)
            mutant = creator.Individual([
                pop[i][j] + F_i * (pbest[j] - pop[i][j]) + F_i * (pop[r1][j] - x_r2[j])
                for j in range(dim)
            ])

            # Binomial crossover
            j_rand = random.randint(0, dim - 1)
            trial = creator.Individual([
                mutant[j] if (random.random() < CR_i or j == j_rand) else pop[i][j]
                for j in range(dim)
            ])

            # Clip to bounds
            for j in range(dim):
                trial[j] = max(lo, min(hi, trial[j]))

            trial.fitness.values = toolbox.evaluate(trial)
            trial_list.append(trial)

        # Selection and success history update
        new_pop = []
        for i in range(N_current):
            trial = trial_list[i]
            if trial.fitness.values[0] <= pop[i].fitness.values[0]:
                if trial.fitness.values[0] < pop[i].fitness.values[0]:
                    S_F.append(F_list[i])
                    S_CR.append(CR_list[i])
                    delta_f.append(abs(pop[i].fitness.values[0] - trial.fitness.values[0]))
                    # Add inferior parent to archive
                    archive.append(creator.Individual(pop[i][:]))
                new_pop.append(trial)
            else:
                new_pop.append(pop[i])

        pop = new_pop

        # Update success history
        if S_F:
            weights = np.array(delta_f)
            weights = weights / (weights.sum() + 1e-30)

            # Weighted Lehmer mean for F
            S_F_arr = np.array(S_F)
            mean_F = np.sum(weights * S_F_arr ** 2) / (np.sum(weights * S_F_arr) + 1e-30)
            M_F[k] = mean_F

            # Weighted arithmetic mean for CR
            S_CR_arr = np.array(S_CR)
            mean_CR = np.sum(weights * S_CR_arr)
            M_CR[k] = mean_CR

            k = (k + 1) % H

        # Trim archive to at most N_current
        while len(archive) > N_current:
            archive.pop(random.randint(0, len(archive) - 1))

        # Linear population size reduction
        N_next = int(round(N_init + (N_min - N_init) * (gen + 1) / n_generations))
        N_next = max(N_min, N_next)
        if N_next < len(pop):
            # Remove worst individuals
            pop.sort(key=lambda ind: ind.fitness.values[0])
            pop = pop[:N_next]

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
