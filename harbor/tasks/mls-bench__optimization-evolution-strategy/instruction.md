# MLS-Bench: optimization-evolution-strategy

# Evolutionary Optimization Strategy Design

## Research Question
Design a novel combination of selection, crossover, and mutation operators (and/or a novel evolutionary loop) for continuous black-box optimization that performs well across a variety of problem landscapes.

## Background
Evolutionary algorithms (EAs) are population-based metaheuristics for black-box optimization. The three core operators — **selection**, **crossover**, and **mutation** — together with the overall evolutionary loop design, determine an EA's performance. Different operator families have strengths on different function landscapes (multimodal, ill-conditioned, high-dimensional), and no single strategy dominates all.

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
- Respect the function signature and return types — the harness below the editable section is fixed.

## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/deap/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — are not permitted.

- `deap/custom_evolution.py`
- editable lines **87–225**

## Readable Context

### `deap/custom_evolution.py`  [EDITABLE — lines 87–225 only]

```python
     1: #!/usr/bin/env python3
     2: """Evolutionary Optimization Strategy scaffold.
     3:
     4: The goal is to minimize a black-box objective function by designing effective
     5: selection, crossover, and mutation operators (and possibly the full
     6: evolutionary loop).
     7: """
     8:
     9: import argparse
    10: import math
    11: import random
    12: import time
    13: from typing import List, Tuple, Callable
    14:
    15: import numpy as np
    16: from deap import base, creator, tools
    17:
    18: # ================================================================
    19: # FIXED — Objective functions and infrastructure (contents withheld)
    20: # ================================================================
    21:
    22: # A dictionary of objective functions (each minimization) and their
    23: # per-dimension bounds is defined here. Function bodies and bounds are
    24: # not visible to the agent.
    25:
    61: # --- DEAP fitness and individual setup ---
    62:
    63: if not hasattr(creator, "FitnessMin"):
    64:     creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    65: if not hasattr(creator, "Individual"):
    66:     creator.create("Individual", list, fitness=creator.FitnessMin)
    67:
    68:
    69: def make_individual(toolbox, dim: int, lo: float, hi: float):
    70:     """Create a random individual within bounds."""
    71:     ind = creator.Individual([random.uniform(lo, hi) for _ in range(dim)])
    72:     return ind
    73:
    74:
    75: def clip_individual(individual, lo: float, hi: float):
    76:     """Clip individual's genes to stay within bounds."""
    77:     for i in range(len(individual)):
    78:         individual[i] = max(lo, min(hi, individual[i]))
    79:     return individual
    80:
    81:
    82: # ================================================================
    83: # EDITABLE SECTION — Design your evolutionary strategy below
    84: # (lines 87 to 225)
    85: # ================================================================
    86:
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
   112:     # Default placeholder; replace with your own recombination operator.
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
   128:     # Default placeholder; replace with your own mutation operator.
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
   228: # FIXED — Harness (do not modify below; contents withheld)
   229: # ================================================================
```

## Tips

- Keep the function/class signatures of the editable regions identical.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.

Good luck.
