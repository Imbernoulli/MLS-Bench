# MLS-Bench: optimization-evolution-strategy

# Evolutionary Optimization Strategy Design

## Research Question
Design a novel combination of selection, crossover, and mutation operators (and/or a novel evolutionary loop) for continuous black-box optimization that outperforms standard approaches across multiple benchmark functions.

## Background
Evolutionary algorithms (EAs) are population-based metaheuristics for black-box optimization. The three core operators — **selection**, **crossover**, and **mutation** — together with the overall evolutionary loop design, determine an EA's performance. Standard approaches include:
- **Genetic Algorithms (GA)** with Tournament selection + Simulated Binary Crossover (SBX) + Polynomial Mutation (Deb and Agrawal, 1995).
- **CMA-ES** — Covariance Matrix Adaptation Evolution Strategy (Hansen and Ostermeier, "Completely Derandomized Self-Adaptation in Evolution Strategies", *Evolutionary Computation* 9(2), 2001).
- **Differential Evolution (DE)** — uses vector differences between population members for mutation (Storn and Price, "Differential Evolution", *J. Global Optim.* 11, 1997).
- **L-SHADE** — Success-History based Adaptive DE with Linear population reduction (Tanabe and Fukunaga, "Improving the Search Performance of SHADE Using Linear Population Size Reduction", IEEE CEC 2014; CEC 2014 winner).

Each has strengths on different function landscapes (multimodal, ill-conditioned, high-dimensional), but no single strategy dominates all.

## Task
Modify the editable section of `custom_evolution.py` to implement a novel or improved evolutionary strategy. You may modify:
- `custom_select(population, k, toolbox)` — selection operator.
- `custom_crossover(ind1, ind2)` — crossover/recombination operator.
- `custom_mutate(individual, lo, hi)` — mutation operator.
- `run_evolution(...)` — the full evolutionary loop (you can restructure the algorithm entirely).

The DEAP library (`deap.base`, `deap.creator`, `deap.tools`) is available. You may also use `numpy`, `scipy`, `math`, and `random`.

## Interface
- **Individuals**: lists of floats, each with a `.fitness.values` attribute (tuple of one float for minimization).
- **`run_evolution`** must return `(best_individual, fitness_history)` where `fitness_history` is a list of best fitness per generation.
- **TRAIN_METRICS**: print `TRAIN_METRICS gen=G best_fitness=F avg_fitness=A` periodically (every 50 generations).
- Respect the function signature and return types — the evaluation harness below the editable section is fixed.

## Baselines (paper-cited reference implementations)
- **ga_sbx** — Genetic Algorithm with Simulated Binary Crossover and Polynomial Mutation (Deb and Agrawal, 1995); paper-default `eta_c = eta_m = 20`, mutation probability `1/n`.
- **de** — Classical DE/rand/1/bin (Storn and Price, 1997); paper-default `F = 0.5`, `CR = 0.9`.
- **lshade** — L-SHADE (Tanabe and Fukunaga, IEEE CEC 2014); paper-default initial population `18 * n`, archive size `2.6 * pop`, history memory `H = 6`, linear population reduction to `N_min = 4`.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/deap/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `deap/custom_evolution.py`
- editable lines **87–225**




## Readable Context


### `deap/custom_evolution.py`  [EDITABLE — lines 87–225 only]

```python
     1: #!/usr/bin/env python3
     2: """Evolutionary Optimization Strategy Benchmark.
     3: 
     4: This script benchmarks an evolutionary optimization strategy on standard
     5: continuous optimization test functions (Rastrigin, Rosenbrock, Ackley).
     6: The goal is to minimize each function by designing effective selection,
     7: crossover, and mutation operators.
     8: 
     9: Usage:
    10:     python deap/custom_evolution.py --function rastrigin --dim 30 --seed 42
    11: """
    12: 
    13: import argparse
    14: import math
    15: import random
    16: import time
    17: from typing import List, Tuple, Callable
    18: 
    19: import numpy as np
    20: from deap import base, creator, tools
    21: 
    22: # ================================================================
    23: # FIXED — Benchmark functions and infrastructure (do not modify)
    24: # ================================================================
    25: 
    26: # --- Benchmark function definitions ---
    27: 
    28: def rastrigin(individual: List[float]) -> Tuple[float]:
    29:     """Rastrigin function. Global minimum: f(0,...,0) = 0. Domain: [-5.12, 5.12]."""
    30:     A = 10.0
    31:     n = len(individual)
    32:     val = A * n + sum(x**2 - A * math.cos(2 * math.pi * x) for x in individual)
    33:     return (val,)
    34: 
    35: 
    36: def rosenbrock(individual: List[float]) -> Tuple[float]:
    37:     """Rosenbrock function. Global minimum: f(1,...,1) = 0. Domain: [-5, 10]."""
    38:     val = sum(
    39:         100.0 * (individual[i + 1] - individual[i]**2)**2 + (1 - individual[i])**2
    40:         for i in range(len(individual) - 1)
    41:     )
    42:     return (val,)
    43: 
    44: 
    45: def ackley(individual: List[float]) -> Tuple[float]:
    46:     """Ackley function. Global minimum: f(0,...,0) = 0. Domain: [-32.768, 32.768]."""
    47:     n = len(individual)
    48:     sum_sq = sum(x**2 for x in individual) / n
    49:     sum_cos = sum(math.cos(2 * math.pi * x) for x in individual) / n
    50:     val = -20.0 * math.exp(-0.2 * math.sqrt(sum_sq)) - math.exp(sum_cos) + 20.0 + math.e
    51:     return (val,)
    52: 
    53: 
    54: BENCHMARKS = {
    55:     "rastrigin": {"func": rastrigin, "bounds": (-5.12, 5.12)},
    56:     "rosenbrock": {"func": rosenbrock, "bounds": (-5.0, 10.0)},
    57:     "ackley": {"func": ackley, "bounds": (-32.768, 32.768)},
    58: }
    59: 
    60: 
    61: # --- DEAP fitness and individual setup ---
    62: 
    63: # Single-objective minimization
    64: if not hasattr(creator, "FitnessMin"):
    65:     creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    66: if not hasattr(creator, "Individual"):
    67:     creator.create("Individual", list, fitness=creator.FitnessMin)
    68: 
    69: 
    70: def make_individual(toolbox, dim: int, lo: float, hi: float):
    71:     """Create a random individual within bounds."""
    72:     ind = creator.Individual([random.uniform(lo, hi) for _ in range(dim)])
    73:     return ind
    74: 
    75: 
    76: def clip_individual(individual, lo: float, hi: float):
    77:     """Clip individual's genes to stay within bounds."""
    78:     for i in range(len(individual)):
    79:         individual[i] = max(lo, min(hi, individual[i]))
    80:     return individual
    81: 
    82: 
    83: # ================================================================
    84: # EDITABLE SECTION — Design your evolutionary strategy below
    85: # (lines 87 to 225)
    86: # ================================================================
    87: 
    88: def custom_select(population: list, k: int, toolbox=None) -> list:
    89:     """Select k individuals from the population.
    90: 
    91:     Args:
    92:         population: List of individuals (each has a .fitness.values attribute).
    93:         k: Number of individuals to select.
    94:         toolbox: The DEAP toolbox (optional, for access to other operators).
    95: 
    96:     Returns:
    97:         List of k selected individuals (deep copies recommended).
    98:     """
    99:     # Default: tournament selection with tournament size 3
   100:     return tools.selTournament(population, k, tournsize=3)
   101: 
   102: 
   103: def custom_crossover(ind1: list, ind2: list) -> Tuple[list, list]:
   104:     """Apply crossover to two individuals.
   105: 
   106:     Args:
   107:         ind1, ind2: Parent individuals (lists of floats).
   108: 
   109:     Returns:
   110:         Tuple of two offspring individuals (modified in-place).
   111:     """
   112:     # Default: simulated binary crossover (SBX), eta=20
   113:     tools.cxSimulatedBinary(ind1, ind2, eta=20.0)
   114:     return ind1, ind2
   115: 
   116: 
   117: def custom_mutate(individual: list, lo: float, hi: float) -> Tuple[list]:
   118:     """Apply mutation to an individual.
   119: 
   120:     Args:
   121:         individual: The individual to mutate (list of floats).
   122:         lo: Lower bound for genes.
   123:         hi: Upper bound for genes.
   124: 
   125:     Returns:
   126:         Tuple containing the mutated individual.
   127:     """
   128:     # Default: polynomial mutation, eta=20, indpb=1/dim
   129:     tools.mutPolynomialBounded(
   130:         individual, eta=20.0, low=lo, up=hi,
   131:         indpb=1.0 / len(individual)
   132:     )
   133:     return (individual,)
   134: 
   135: 
   136: def run_evolution(
   137:     evaluate_func: Callable,
   138:     dim: int,
   139:     lo: float,
   140:     hi: float,
   141:     pop_size: int,
   142:     n_generations: int,
   143:     cx_prob: float,
   144:     mut_prob: float,
   145:     seed: int,
   146: ) -> Tuple[list, list]:
   147:     """Run the evolutionary algorithm.
   148: 
   149:     Args:
   150:         evaluate_func: Fitness function mapping individual -> (fitness_value,).
   151:         dim: Dimensionality of the search space.
   152:         lo: Lower bound for each dimension.
   153:         hi: Upper bound for each dimension.
   154:         pop_size: Population size.
   155:         n_generations: Number of generations.
   156:         cx_prob: Crossover probability.
   157:         mut_prob: Mutation probability.
   158:         seed: Random seed.
   159: 
   160:     Returns:
   161:         best_individual: The best individual found.
   162:         fitness_history: List of best fitness per generation.
   163:     """
   164:     random.seed(seed)
   165:     np.random.seed(seed)
   166: 
   167:     # Setup toolbox
   168:     toolbox = base.Toolbox()
   169:     toolbox.register("individual", make_individual, toolbox, dim, lo, hi)
   170:     toolbox.register("population", tools.initRepeat, list, toolbox.individual)
   171:     toolbox.register("evaluate", evaluate_func)
   172: 
   173:     # Initialize population
   174:     pop = toolbox.population(n=pop_size)
   175:     fitnesses = list(map(toolbox.evaluate, pop))
   176:     for ind, fit in zip(pop, fitnesses):
   177:         ind.fitness.values = fit
   178: 
   179:     fitness_history = []
   180: 
   181:     for gen in range(n_generations):
   182:         # Selection
   183:         offspring = custom_select(pop, len(pop), toolbox)
   184:         offspring = [toolbox.clone(ind) for ind in offspring]
   185: 
   186:         # Crossover
   187:         for i in range(0, len(offspring) - 1, 2):
   188:             if random.random() < cx_prob:
   189:                 custom_crossover(offspring[i], offspring[i + 1])
   190:                 del offspring[i].fitness.values
   191:                 del offspring[i + 1].fitness.values
   192: 
   193:         # Mutation
   194:         for i in range(len(offspring)):
   195:             if random.random() < mut_prob:
   196:                 custom_mutate(offspring[i], lo, hi)
   197:                 del offspring[i].fitness.values
   198: 
   199:         # Clip to bounds
   200:         for ind in offspring:
   201:             clip_individual(ind, lo, hi)
   202: 
   203:         # Evaluate individuals with invalid fitness
   204:         invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
   205:         fitnesses = list(map(toolbox.evaluate, invalid_ind))
   206:         for ind, fit in zip(invalid_ind, fitnesses):
   207:             ind.fitness.values = fit
   208: 
   209:         # Replace population
   210:         pop[:] = offspring
   211: 
   212:         # Track best fitness
   213:         best_fit = min(ind.fitness.values[0] for ind in pop)
   214:         fitness_history.append(best_fit)
   215: 
   216:         if (gen + 1) % 50 == 0 or gen == 0:
   217:             avg_fit = sum(ind.fitness.values[0] for ind in pop) / len(pop)
   218:             print(
   219:                 f"TRAIN_METRICS gen={gen+1} best_fitness={best_fit:.6e} "
   220:                 f"avg_fitness={avg_fit:.6e}",
   221:                 flush=True,
   222:             )
   223: 
   224:     best_ind = min(pop, key=lambda ind: ind.fitness.values[0])
   225:     return best_ind, fitness_history
   226: 
   227: # ================================================================
   228: # FIXED — Evaluation harness (do not modify below)
   229: # ================================================================
   230: 
   231: 
   232: def compute_convergence_gen(fitness_history: list, threshold_ratio: float = 0.01) -> int:
   233:     """Compute the generation at which fitness first reaches within threshold of final best.
   234: 
   235:     Returns the 1-indexed generation number, or len(fitness_history) if never converged.
   236:     """
   237:     if not fitness_history:
   238:         return 0
   239:     final_best = fitness_history[-1]
   240:     # threshold: within 1% of final best, or absolute 1e-6 for near-zero
   241:     threshold = max(abs(final_best) * threshold_ratio, 1e-6)
   242:     for i, f in enumerate(fitness_history):
   243:         if abs(f - final_best) <= threshold:
   244:             return i + 1
   245:     return len(fitness_history)
   246: 
   247: 
   248: def main():
   249:     parser = argparse.ArgumentParser(description="Evolutionary Optimization Benchmark")
   250:     parser.add_argument("--function", type=str, required=True,
   251:                         choices=list(BENCHMARKS.keys()),
   252:                         help="Benchmark function to optimize")
   253:     parser.add_argument("--dim", type=int, default=30,
   254:                         help="Dimensionality of the search space (default: 30)")
   255:     parser.add_argument("--pop-size", type=int, default=200,
   256:                         help="Population size (default: 200)")
   257:     parser.add_argument("--n-generations", type=int, default=500,
   258:                         help="Number of generations (default: 500)")
   259:     parser.add_argument("--cx-prob", type=float, default=0.9,
   260:                         help="Crossover probability (default: 0.9)")
   261:     parser.add_argument("--mut-prob", type=float, default=0.2,
   262:                         help="Mutation probability (default: 0.2)")
   263:     parser.add_argument("--seed", type=int, default=42,
   264:                         help="Random seed")
   265:     args = parser.parse_args()
   266: 
   267:     bench = BENCHMARKS[args.function]
   268:     evaluate_func = bench["func"]
   269:     lo, hi = bench["bounds"]
   270: 
   271:     print(f"=== {args.function.upper()} (dim={args.dim}) ===", flush=True)
   272:     print(f"Bounds: [{lo}, {hi}], Pop: {args.pop_size}, Gens: {args.n_generations}", flush=True)
   273: 
   274:     t0 = time.time()
   275:     best_ind, fitness_history = run_evolution(
   276:         evaluate_func=evaluate_func,
   277:         dim=args.dim,
   278:         lo=lo,
   279:         hi=hi,
   280:         pop_size=args.pop_size,
   281:         n_generations=args.n_generations,
   282:         cx_prob=args.cx_prob,
   283:         mut_prob=args.mut_prob,
   284:         seed=args.seed,
   285:     )
   286:     elapsed = time.time() - t0
   287: 
   288:     best_fitness = best_ind.fitness.values[0]
   289:     convergence_gen = compute_convergence_gen(fitness_history)
   290: 
   291:     print(f"\n=== Results ===", flush=True)
   292:     print(f"Best fitness: {best_fitness:.6e}", flush=True)
   293:     print(f"Convergence generation: {convergence_gen}/{args.n_generations}", flush=True)
   294:     print(f"Wall time: {elapsed:.1f}s", flush=True)
   295:     print(
   296:         f"TEST_METRICS best_fitness={best_fitness:.6e} "
   297:         f"convergence_gen={convergence_gen}",
   298:         flush=True,
   299:     )
   300: 
   301: 
   302: if __name__ == "__main__":
   303:     main()
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `ga_sbx` baseline — editable region  [READ-ONLY — reference implementation]

In `deap/custom_evolution.py`:

```python
Lines 87–172:
    84: # EDITABLE SECTION — Design your evolutionary strategy below
    85: # (lines 87 to 225)
    86: # ================================================================
    87: 
    88: def custom_select(population: list, k: int, toolbox=None) -> list:
    89:     """Tournament selection with tournament size 3."""
    90:     return tools.selTournament(population, k, tournsize=3)
    91: 
    92: 
    93: def custom_crossover(ind1: list, ind2: list) -> Tuple[list, list]:
    94:     """Simulated Binary Crossover (SBX), eta=20."""
    95:     tools.cxSimulatedBinary(ind1, ind2, eta=20.0)
    96:     return ind1, ind2
    97: 
    98: 
    99: def custom_mutate(individual: list, lo: float, hi: float) -> Tuple[list]:
   100:     """Polynomial bounded mutation, eta=20, indpb=1/dim."""
   101:     tools.mutPolynomialBounded(
   102:         individual, eta=20.0, low=lo, up=hi,
   103:         indpb=1.0 / len(individual)
   104:     )
   105:     return (individual,)
   106: 
   107: 
   108: def run_evolution(
   109:     evaluate_func: Callable,
   110:     dim: int,
   111:     lo: float,
   112:     hi: float,
   113:     pop_size: int,
   114:     n_generations: int,
   115:     cx_prob: float,
   116:     mut_prob: float,
   117:     seed: int,
   118: ) -> Tuple[list, list]:
   119:     """Standard GA with tournament selection, SBX crossover, polynomial mutation."""
   120:     random.seed(seed)
   121:     np.random.seed(seed)
   122: 
   123:     toolbox = base.Toolbox()
   124:     toolbox.register("individual", make_individual, toolbox, dim, lo, hi)
   125:     toolbox.register("population", tools.initRepeat, list, toolbox.individual)
   126:     toolbox.register("evaluate", evaluate_func)
   127: 
   128:     pop = toolbox.population(n=pop_size)
   129:     fitnesses = list(map(toolbox.evaluate, pop))
   130:     for ind, fit in zip(pop, fitnesses):
   131:         ind.fitness.values = fit
   132: 
   133:     fitness_history = []
   134: 
   135:     for gen in range(n_generations):
   136:         offspring = custom_select(pop, len(pop), toolbox)
   137:         offspring = [toolbox.clone(ind) for ind in offspring]
   138: 
   139:         for i in range(0, len(offspring) - 1, 2):
   140:             if random.random() < cx_prob:
   141:                 custom_crossover(offspring[i], offspring[i + 1])
   142:                 del offspring[i].fitness.values
   143:                 del offspring[i + 1].fitness.values
   144: 
   145:         for i in range(len(offspring)):
   146:             if random.random() < mut_prob:
   147:                 custom_mutate(offspring[i], lo, hi)
   148:                 del offspring[i].fitness.values
   149: 
   150:         for ind in offspring:
   151:             clip_individual(ind, lo, hi)
   152: 
   153:         invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
   154:         fitnesses = list(map(toolbox.evaluate, invalid_ind))
   155:         for ind, fit in zip(invalid_ind, fitnesses):
   156:             ind.fitness.values = fit
   157: 
   158:         pop[:] = offspring
   159: 
   160:         best_fit = min(ind.fitness.values[0] for ind in pop)
   161:         fitness_history.append(best_fit)
   162: 
   163:         if (gen + 1) % 50 == 0 or gen == 0:
   164:             avg_fit = sum(ind.fitness.values[0] for ind in pop) / len(pop)
   165:             print(
   166:                 f"TRAIN_METRICS gen={gen+1} best_fitness={best_fit:.6e} "
   167:                 f"avg_fitness={avg_fit:.6e}",
   168:                 flush=True,
   169:             )
   170: 
   171:     best_ind = min(pop, key=lambda ind: ind.fitness.values[0])
   172:     return best_ind, fitness_history
   173: 
   174: # ================================================================
   175: # FIXED — Evaluation harness (do not modify below)
```

### `de` baseline — editable region  [READ-ONLY — reference implementation]

In `deap/custom_evolution.py`:

```python
Lines 87–181:
    84: # EDITABLE SECTION — Design your evolutionary strategy below
    85: # (lines 87 to 225)
    86: # ================================================================
    87: 
    88: def custom_select(population: list, k: int, toolbox=None) -> list:
    89:     """Not used directly in DE (greedy selection is built into run_evolution)."""
    90:     return population[:k]
    91: 
    92: 
    93: def custom_crossover(ind1: list, ind2: list) -> Tuple[list, list]:
    94:     """Not used directly in DE (binomial crossover is built into run_evolution)."""
    95:     return ind1, ind2
    96: 
    97: 
    98: def custom_mutate(individual: list, lo: float, hi: float) -> Tuple[list]:
    99:     """Not used directly in DE (DE mutation is built into run_evolution)."""
   100:     return (individual,)
   101: 
   102: 
   103: def run_evolution(
   104:     evaluate_func: Callable,
   105:     dim: int,
   106:     lo: float,
   107:     hi: float,
   108:     pop_size: int,
   109:     n_generations: int,
   110:     cx_prob: float,
   111:     mut_prob: float,
   112:     seed: int,
   113: ) -> Tuple[list, list]:
   114:     """Differential Evolution: DE/rand/1/bin.
   115: 
   116:     F (scale factor) = 0.5, CR (crossover rate) = 0.9.
   117:     Greedy selection: trial replaces target only if fitness improves.
   118:     """
   119:     random.seed(seed)
   120:     np.random.seed(seed)
   121: 
   122:     F = 0.5   # Scale factor (Storn & Price recommended default)
   123:     CR = 0.9  # Crossover rate
   124: 
   125:     toolbox = base.Toolbox()
   126:     toolbox.register("individual", make_individual, toolbox, dim, lo, hi)
   127:     toolbox.register("population", tools.initRepeat, list, toolbox.individual)
   128:     toolbox.register("evaluate", evaluate_func)
   129: 
   130:     pop = toolbox.population(n=pop_size)
   131:     fitnesses = list(map(toolbox.evaluate, pop))
   132:     for ind, fit in zip(pop, fitnesses):
   133:         ind.fitness.values = fit
   134: 
   135:     fitness_history = []
   136: 
   137:     for gen in range(n_generations):
   138:         for i in range(len(pop)):
   139:             # Select three distinct random individuals (not i)
   140:             candidates = list(range(len(pop)))
   141:             candidates.remove(i)
   142:             r1, r2, r3 = random.sample(candidates, 3)
   143:             x_r1, x_r2, x_r3 = pop[r1], pop[r2], pop[r3]
   144: 
   145:             # DE/rand/1 mutation
   146:             mutant = creator.Individual([
   147:                 x_r1[j] + F * (x_r2[j] - x_r3[j])
   148:                 for j in range(dim)
   149:             ])
   150: 
   151:             # Binomial crossover
   152:             j_rand = random.randint(0, dim - 1)
   153:             trial = creator.Individual([
   154:                 mutant[j] if (random.random() < CR or j == j_rand) else pop[i][j]
   155:                 for j in range(dim)
   156:             ])
   157: 
   158:             # Clip to bounds
   159:             for j in range(dim):
   160:                 trial[j] = max(lo, min(hi, trial[j]))
   161: 
   162:             # Evaluate trial
   163:             trial.fitness.values = toolbox.evaluate(trial)
   164: 
   165:             # Greedy selection
   166:             if trial.fitness.values[0] <= pop[i].fitness.values[0]:
   167:                 pop[i] = trial
   168: 
   169:         best_fit = min(ind.fitness.values[0] for ind in pop)
   170:         fitness_history.append(best_fit)
   171: 
   172:         if (gen + 1) % 50 == 0 or gen == 0:
   173:             avg_fit = sum(ind.fitness.values[0] for ind in pop) / len(pop)
   174:             print(
   175:                 f"TRAIN_METRICS gen={gen+1} best_fitness={best_fit:.6e} "
   176:                 f"avg_fitness={avg_fit:.6e}",
   177:                 flush=True,
   178:             )
   179: 
   180:     best_ind = min(pop, key=lambda ind: ind.fitness.values[0])
   181:     return best_ind, fitness_history
   182: 
   183: # ================================================================
   184: # FIXED — Evaluation harness (do not modify below)
```

### `lshade` baseline — editable region  [READ-ONLY — reference implementation]

In `deap/custom_evolution.py`:

```python
Lines 87–287:
    84: # EDITABLE SECTION — Design your evolutionary strategy below
    85: # (lines 87 to 225)
    86: # ================================================================
    87: 
    88: def custom_select(population: list, k: int, toolbox=None) -> list:
    89:     """Not used in L-SHADE (adaptive DE handles selection internally)."""
    90:     return population[:k]
    91: 
    92: 
    93: def custom_crossover(ind1: list, ind2: list) -> Tuple[list, list]:
    94:     """Not used in L-SHADE (binomial crossover built into run_evolution)."""
    95:     return ind1, ind2
    96: 
    97: 
    98: def custom_mutate(individual: list, lo: float, hi: float) -> Tuple[list]:
    99:     """Not used in L-SHADE (adaptive mutation built into run_evolution)."""
   100:     return (individual,)
   101: 
   102: 
   103: def run_evolution(
   104:     evaluate_func: Callable,
   105:     dim: int,
   106:     lo: float,
   107:     hi: float,
   108:     pop_size: int,
   109:     n_generations: int,
   110:     cx_prob: float,
   111:     mut_prob: float,
   112:     seed: int,
   113: ) -> Tuple[list, list]:
   114:     """L-SHADE: Success-History based Adaptive DE with Linear Population Reduction.
   115: 
   116:     - Adaptive F (Cauchy) and CR (Normal) from success history
   117:     - current-to-pbest/1 mutation with external archive
   118:     - Linear population size reduction from N_init to N_min
   119:     """
   120:     random.seed(seed)
   121:     np.random.seed(seed)
   122: 
   123:     # --- Hyperparameters (Tanabe & Fukunaga, CEC 2014) ---
   124:     # The paper recommends N_init = 18·D, but on small fixed budgets (as in
   125:     # our 400 pop × 1000 gen setting) that value starves the search of
   126:     # generations: on Rastrigin-100D, N_init=1800 with matched total-eval
   127:     # budget degraded from 128 → 313. Use pop_size as given and the
   128:     # canonical N_min = 4 (paper §III-B), which lets the linear population
   129:     # reduction actually run. Budget stays identical to CMA-ES/DE/GA.
   130:     H = 6  # History size (paper default)
   131:     N_init = pop_size
   132:     N_min = 4  # Minimum population size
   133:     p_min = 2.0 / N_init  # Minimum p for pbest
   134:     p_max = 0.2  # Maximum p for pbest
   135: 
   136:     toolbox = base.Toolbox()
   137:     toolbox.register("individual", make_individual, toolbox, dim, lo, hi)
   138:     toolbox.register("population", tools.initRepeat, list, toolbox.individual)
   139:     toolbox.register("evaluate", evaluate_func)
   140: 
   141:     # Initialize population
   142:     pop = toolbox.population(n=N_init)
   143:     fitnesses = list(map(toolbox.evaluate, pop))
   144:     for ind, fit in zip(pop, fitnesses):
   145:         ind.fitness.values = fit
   146: 
   147:     # Success history for F and CR
   148:     M_F = [0.5] * H
   149:     M_CR = [0.5] * H
   150:     k = 0  # History index
   151: 
   152:     # External archive of inferior solutions
   153:     archive = []
   154: 
   155:     fitness_history = []
   156: 
   157:     for gen in range(n_generations):
   158:         N_current = len(pop)
   159: 
   160:         # Collect successful F and CR values and their fitness improvements
   161:         S_F = []
   162:         S_CR = []
   163:         delta_f = []  # fitness improvement for weighting
   164: 
   165:         trial_list = []
   166:         F_list = []
   167:         CR_list = []
   168: 
   169:         for i in range(N_current):
   170:             # Sample F from Cauchy(M_F[r], 0.1), truncate to (0, 1]
   171:             r = random.randint(0, H - 1)
   172:             while True:
   173:                 F_i = M_F[r] + 0.1 * np.random.standard_cauchy()
   174:                 if F_i > 0:
   175:                     break
   176:             F_i = min(F_i, 1.0)
   177: 
   178:             # Sample CR from Normal(M_CR[r], 0.1), clamp to [0, 1]
   179:             CR_i = np.random.normal(M_CR[r], 0.1)
   180:             CR_i = max(0.0, min(1.0, CR_i))
   181: 
   182:             F_list.append(F_i)
   183:             CR_list.append(CR_i)
   184: 
   185:             # current-to-pbest/1 mutation
   186:             # Choose p_i uniformly from [p_min, p_max]
   187:             p_i = random.uniform(p_min, p_max)
   188:             n_pbest = max(1, int(round(p_i * N_current)))
   189:             sorted_pop = sorted(pop, key=lambda ind: ind.fitness.values[0])
   190:             pbest = random.choice(sorted_pop[:n_pbest])
   191: 
   192:             # Select r1 from pop (r1 != i)
   193:             candidates = list(range(N_current))
   194:             candidates.remove(i)
   195:             r1 = random.choice(candidates)
   196: 
   197:             # Select r2 from pop + archive (r2 != i and r2 != r1)
   198:             union = list(range(N_current + len(archive)))
   199:             union_exclude = {i, r1}
   200:             union_avail = [x for x in union if x not in union_exclude]
   201:             if not union_avail:
   202:                 union_avail = [x for x in union if x != i]
   203:             r2_idx = random.choice(union_avail)
   204:             if r2_idx < N_current:
   205:                 x_r2 = pop[r2_idx]
   206:             else:
   207:                 x_r2 = archive[r2_idx - N_current]
   208: 
   209:             # Mutation: v = x_i + F * (pbest - x_i) + F * (x_r1 - x_r2)
   210:             mutant = creator.Individual([
   211:                 pop[i][j] + F_i * (pbest[j] - pop[i][j]) + F_i * (pop[r1][j] - x_r2[j])
   212:                 for j in range(dim)
   213:             ])
   214: 
   215:             # Binomial crossover
   216:             j_rand = random.randint(0, dim - 1)
   217:             trial = creator.Individual([
   218:                 mutant[j] if (random.random() < CR_i or j == j_rand) else pop[i][j]
   219:                 for j in range(dim)
   220:             ])
   221: 
   222:             # Clip to bounds
   223:             for j in range(dim):
   224:                 trial[j] = max(lo, min(hi, trial[j]))
   225: 
   226:             trial.fitness.values = toolbox.evaluate(trial)
   227:             trial_list.append(trial)
   228: 
   229:         # Selection and success history update
   230:         new_pop = []
   231:         for i in range(N_current):
   232:             trial = trial_list[i]
   233:             if trial.fitness.values[0] <= pop[i].fitness.values[0]:
   234:                 if trial.fitness.values[0] < pop[i].fitness.values[0]:
   235:                     S_F.append(F_list[i])
   236:                     S_CR.append(CR_list[i])
   237:                     delta_f.append(abs(pop[i].fitness.values[0] - trial.fitness.values[0]))
   238:                     # Add inferior parent to archive
   239:                     archive.append(creator.Individual(pop[i][:]))
   240:                 new_pop.append(trial)
   241:             else:
   242:                 new_pop.append(pop[i])
   243: 
   244:         pop = new_pop
   245: 
   246:         # Update success history
   247:         if S_F:
   248:             weights = np.array(delta_f)
   249:             weights = weights / (weights.sum() + 1e-30)
   250: 
   251:             # Weighted Lehmer mean for F
   252:             S_F_arr = np.array(S_F)
   253:             mean_F = np.sum(weights * S_F_arr ** 2) / (np.sum(weights * S_F_arr) + 1e-30)
   254:             M_F[k] = mean_F
   255: 
   256:             # Weighted arithmetic mean for CR
   257:             S_CR_arr = np.array(S_CR)
   258:             mean_CR = np.sum(weights * S_CR_arr)
   259:             M_CR[k] = mean_CR
   260: 
   261:             k = (k + 1) % H
   262: 
   263:         # Trim archive to at most N_current
   264:         while len(archive) > N_current:
   265:             archive.pop(random.randint(0, len(archive) - 1))
   266: 
   267:         # Linear population size reduction
   268:         N_next = int(round(N_init + (N_min - N_init) * (gen + 1) / n_generations))
   269:         N_next = max(N_min, N_next)
   270:         if N_next < len(pop):
   271:             # Remove worst individuals
   272:             pop.sort(key=lambda ind: ind.fitness.values[0])
   273:             pop = pop[:N_next]
   274: 
   275:         best_fit = min(ind.fitness.values[0] for ind in pop)
   276:         fitness_history.append(best_fit)
   277: 
   278:         if (gen + 1) % 50 == 0 or gen == 0:
   279:             avg_fit = sum(ind.fitness.values[0] for ind in pop) / len(pop)
   280:             print(
   281:                 f"TRAIN_METRICS gen={gen+1} best_fitness={best_fit:.6e} "
   282:                 f"avg_fitness={avg_fit:.6e}",
   283:                 flush=True,
   284:             )
   285: 
   286:     best_ind = min(pop, key=lambda ind: ind.fitness.values[0])
   287:     return best_ind, fitness_history
   288: 
   289: # ================================================================
   290: # FIXED — Evaluation harness (do not modify below)
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
