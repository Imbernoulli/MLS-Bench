# MLS-Bench: ml-symbolic-regression

# Symbolic Regression: GP Search Strategy

## Research Question
Design a genetic-programming search strategy for symbolic regression that more reliably discovers symbolic expressions fitting the target function. The contribution is the *search strategy itself*: fitness shaping, parent selection, crossover/mutation operators, elitism, parsimony pressure, diversity maintenance, or adaptive operator rates.

## Background
Symbolic regression searches the space of mathematical expressions for one that fits observed `(X, y)` data. Genetic programming (GP) maintains a population of expression trees and evolves them by selection, crossover, and mutation. The central tensions are exploration vs. exploitation, controlling expression complexity (bloat), and avoiding premature convergence to local optima.

## Implementation Contract
The agent edits `gplearn/custom_sr.py` and provides four functions:

```python
def fitness_function(tree, X, y) -> float:
    # Lower is better.
    ...

def selection(population, fitnesses, n_select, tournament_size=7) -> list:
    # Return n_select selected individuals (copies).
    ...

def crossover(parent1, parent2, n_features, max_depth=17):
    # Return a new offspring expression tree.
    ...

def mutation(parent, n_features, max_depth=17):
    # Return a new mutated expression tree.
    ...

def evolve_one_generation(population, fitnesses, X_train, y_train,
                          n_features, pop_size,
                          crossover_rate=0.9, mutation_rate=0.05,
                          max_depth=17) -> list:
    # Return the next-generation population (length pop_size).
    ...
```

Available helpers from the skeleton: `safe_evaluate(tree, X)`, `generate_tree('grow'|'full', max_depth, n_features)`, `Tree.copy/size/depth/get_all_nodes()`. Reference code that may be read for context: `gplearn/gplearn/genetic.py`, `gplearn/gplearn/_program.py`, `gplearn/gplearn/fitness.py`. The output must remain an executable symbolic expression, not a black-box predictor.

## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/gplearn/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — are not permitted.

- `gplearn/custom_sr.py`
- editable lines **228–306**

Other files you may **read** for context (do not modify):
- `gplearn/gplearn/genetic.py`
- `gplearn/gplearn/_program.py`
- `gplearn/gplearn/fitness.py`

## Readable Context

### `gplearn/custom_sr.py`  [EDITABLE — lines 228–306 only]

```python
     1: #!/usr/bin/env python3
     2: """Symbolic Regression via Genetic Programming.
     3:
     4: A self-contained GP framework for symbolic regression.
     5: The editable section contains the search strategy: fitness function,
     6: selection, crossover, mutation, and per-generation evolution logic.
     7: """
     8:
     9: import argparse
    10: import math
    11: import random
    12: import sys
    13: import os
    14: import numpy as np
    15:
    16:
    17: # ============================================================
    18: # Operator Definitions (FIXED)
    19: # ============================================================
    20:
    21: def protected_div(a, b):
    22:     """Protected division: returns 1.0 when divisor is near zero."""
    23:     return np.where(np.abs(b) > 1e-10, a / b, 1.0)
    24:
    25:
    26: def protected_log(a):
    27:     """Protected log: returns 0.0 for non-positive inputs."""
    28:     return np.where(np.abs(a) > 1e-10, np.log(np.abs(a)), 0.0)
    29:
    30:
    31: def protected_exp(a):
    32:     """Protected exp: clips input to prevent overflow."""
    33:     return np.exp(np.clip(a, -10, 10))
    34:
    35:
    36: OPERATORS = {
    37:     'add': (np.add, 2),
    38:     'sub': (np.subtract, 2),
    39:     'mul': (np.multiply, 2),
    40:     'div': (protected_div, 2),
    41:     'sin': (np.sin, 1),
    42:     'cos': (np.cos, 1),
    43:     'log': (protected_log, 1),
    44:     'exp': (protected_exp, 1),
    45: }
    46:
    47: OPERATOR_NAMES = list(OPERATORS.keys())
    48:
    49:
    50: # ============================================================
    51: # Tree Representation (FIXED)
    52: # ============================================================
    53:
   (... Node class with evaluate / size / depth / copy / get_all_nodes ...)
   114:
   115: def random_terminal(n_features, const_range=(-5.0, 5.0)):
   116:     """Generate a random terminal node (variable or constant)."""
   117:     ...
   118:
   124: def generate_tree(method, max_depth, n_features, depth=0):
   125:     """Generate a random expression tree using 'grow' or 'full' method."""
   126:     ...
   133:
   135: def ramped_half_and_half(pop_size, max_depth, n_features):
   136:     """Initialize population with ramped half-and-half."""
   137:     ...
   142:
   143:
   144: # ============================================================
   145: # Target Problems (FIXED, contents withheld)
   146: # ============================================================
   147: # A dictionary of target functions with associated sampling ranges and
   148: # train/test sizes is defined here. The agent does not see these.
   199:
   200:
   201: # ============================================================
   202: # Evaluation Utilities (FIXED)
   203: # ============================================================
   204:
   205: def safe_evaluate(tree, X):
   206:     """Evaluate tree with error handling."""
   207:     try:
   208:         result = tree.evaluate(X)
   209:         result = np.nan_to_num(result, nan=1e10, posinf=1e10, neginf=-1e10)
   210:         return np.clip(result, -1e10, 1e10)
   211:     except Exception:
   212:         return np.full(X.shape[0], 1e10)
   213:
   222:
   223:
   224: # ============================================================
   225: # Search Strategy (EDITABLE)
   226: # ============================================================
   227:
   228: def fitness_function(tree, X, y):
   229:     """Evaluate fitness of a candidate program. Lower is better."""
   230:     y_pred = safe_evaluate(tree, X)
   231:     return float(np.mean((y - y_pred) ** 2))
   232:
   233:
   234: def selection(population, fitnesses, n_select):
   235:     """Select individuals from the population for reproduction.
   236:
   237:     Args:
   238:         population: list of Node trees
   239:         fitnesses: list of float fitness values (lower is better)
   240:         n_select: int number of individuals to select
   241:
   242:     Returns:
   243:         list of Node copies of selected individuals
   244:     """
   245:     selected = []
   246:     for _ in range(n_select):
   247:         idx = random.randint(0, len(population) - 1)
   248:         selected.append(population[idx].copy())
   249:     return selected
   250:
   251:
   252: def crossover(parent1, parent2, n_features, max_depth=17):
   253:     """Perform crossover between two parent trees.
   254:
   255:     Returns:
   256:         Node - offspring tree
   257:     """
   258:     return parent1.copy()
   259:
   260:
   261: def mutation(parent, n_features, max_depth=17):
   262:     """Perform mutation on a parent tree.
   263:
   264:     Returns:
   265:         Node - mutated tree
   266:     """
   267:     return parent.copy()
   268:
   269:
   270: def evolve_one_generation(population, fitnesses, X_train, y_train,
   271:                           n_features, pop_size,
   272:                           crossover_rate=0.9, mutation_rate=0.05,
   273:                           max_depth=17):
   274:     """Create the next generation from the current population.
   275:
   276:     Args:
   277:         population: list of Node trees
   278:         fitnesses: list of float fitness values (lower is better)
   279:         X_train: numpy array (n_samples, n_features) - training inputs
   280:         y_train: numpy array (n_samples,) - training targets
   281:         n_features: number of input features
   282:         pop_size: desired population size
   283:         crossover_rate: probability of crossover
   284:         mutation_rate: probability of mutation
   285:         max_depth: maximum allowed tree depth
   286:
   287:     Returns:
   288:         list of Node - next generation population
   289:     """
   290:     new_population = []
   291:     # Elitism: keep best individual
   292:     elite_idx = int(np.argmin(fitnesses))
   293:     new_population.append(population[elite_idx].copy())
   294:
   295:     while len(new_population) < pop_size:
   296:         parents = selection(population, fitnesses, 2)
   297:         r = random.random()
   298:         if r < crossover_rate:
   299:             child = crossover(parents[0], parents[1], n_features, max_depth)
   300:         elif r < crossover_rate + mutation_rate:
   301:             child = mutation(parents[0], n_features, max_depth)
   302:         else:
   303:             child = parents[0]
   304:         new_population.append(child)
   305:
   306:     return new_population[:pop_size]
   307:
   308:
   309: # ============================================================
   310: # Main GP Loop (FIXED)
   311: # ============================================================
   (... argument parsing, generation loop, final reporting on a held-out set ...)
```

## Tips

- Keep the function/class signatures of the editable regions identical.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.

Good luck.
