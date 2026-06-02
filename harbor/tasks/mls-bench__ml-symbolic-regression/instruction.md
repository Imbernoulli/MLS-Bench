# MLS-Bench: ml-symbolic-regression

# Symbolic Regression: GP Search Strategy

## Research Question
Design a genetic-programming search strategy for symbolic regression that more reliably discovers symbolic expressions fitting the target function. The contribution is the *search strategy itself*: fitness shaping, parent selection, crossover/mutation operators, elitism, parsimony pressure, diversity maintenance, or adaptive operator rates.

## Background
Symbolic regression searches the space of mathematical expressions for one that fits observed `(X, y)` data. Genetic programming (GP) maintains a population of expression trees and evolves them by selection, crossover, and mutation. The central tensions are exploration vs. exploitation, controlling expression complexity (bloat), and avoiding premature convergence to local optima.

Reference baselines (provided as `edit_ops` over the same `custom_sr.py` skeleton):
- **Standard GP** — Koza, *Genetic Programming*, MIT Press 1992. Tournament selection (default tournament size 7), subtree crossover (rate 0.9), subtree mutation (rate 0.05), raw MSE fitness, elitism = 1 best individual, max tree depth 17.
- **Parsimony GP** — adds a length penalty: fitness becomes `MSE + alpha * tree_size`. Reference: Poli & McPhee, "Parsimony Pressure Made Easy", GECCO 2008 ([proceedings](https://dl.acm.org/doi/10.1145/1389095.1389340)).
- **Lexicase GP** — Spector 2012; for symbolic regression typically ε-lexicase: La Cava, Spector, Danai, "ε-Lexicase Selection for Regression", GECCO 2016. Selects parents by filtering candidates on randomly ordered training cases, keeping only those within ε of the best on each case.

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

## Fixed Pipeline
The datasets, train/test splits, training loop, and evaluation harness are all fixed and provided by the scaffold.

## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/gplearn/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

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
     4: A self-contained GP framework for symbolic regression benchmarks.
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
    54: class Node:
    55:     """A node in the GP expression tree."""
    56:     __slots__ = ('value', 'children')
    57: 
    58:     def __init__(self, value, children=None):
    59:         self.value = value
    60:         self.children = children or []
    61: 
    62:     @property
    63:     def is_terminal(self):
    64:         return len(self.children) == 0
    65: 
    66:     def evaluate(self, X):
    67:         """Evaluate expression tree on input array X (n_samples, n_features)."""
    68:         if self.is_terminal:
    69:             if isinstance(self.value, str) and self.value.startswith('x'):
    70:                 idx = int(self.value[1:])
    71:                 return X[:, idx].copy()
    72:             else:
    73:                 return np.full(X.shape[0], float(self.value))
    74:         func, arity = OPERATORS[self.value]
    75:         args = [child.evaluate(X) for child in self.children]
    76:         result = func(*args)
    77:         return np.clip(result, -1e15, 1e15)
    78: 
    79:     def size(self):
    80:         """Count total nodes in the tree."""
    81:         return 1 + sum(c.size() for c in self.children)
    82: 
    83:     def depth(self):
    84:         """Compute tree depth."""
    85:         if not self.children:
    86:             return 0
    87:         return 1 + max(c.depth() for c in self.children)
    88: 
    89:     def copy(self):
    90:         """Deep copy the tree."""
    91:         return Node(self.value, [c.copy() for c in self.children])
    92: 
    93:     def get_all_nodes(self):
    94:         """Return a list of (node, parent, child_index) via preorder traversal."""
    95:         result = [(self, None, None)]
    96:         for i, child in enumerate(self.children):
    97:             child_nodes = child.get_all_nodes()
    98:             # Update parent info for direct children
    99:             child_nodes[0] = (child, self, i)
   100:             result.extend(child_nodes)
   101:         return result
   102: 
   103:     def __str__(self):
   104:         if self.is_terminal:
   105:             return str(self.value)
   106:         if len(self.children) == 1:
   107:             return f"{self.value}({self.children[0]})"
   108:         return f"({self.children[0]} {self.value} {self.children[1]})"
   109: 
   110: 
   111: # ============================================================
   112: # Tree Generation (FIXED)
   113: # ============================================================
   114: 
   115: def random_terminal(n_features, const_range=(-5.0, 5.0)):
   116:     """Generate a random terminal node (variable or constant)."""
   117:     if random.random() < 0.5:
   118:         idx = random.randint(0, n_features - 1)
   119:         return Node(f'x{idx}')
   120:     else:
   121:         return Node(str(round(random.uniform(*const_range), 2)))
   122: 
   123: 
   124: def generate_tree(method, max_depth, n_features, depth=0):
   125:     """Generate a random expression tree using 'grow' or 'full' method."""
   126:     if depth >= max_depth or (method == 'grow' and depth > 0 and random.random() < 0.3):
   127:         return random_terminal(n_features)
   128:     op_name = random.choice(OPERATOR_NAMES)
   129:     _, arity = OPERATORS[op_name]
   130:     children = [generate_tree(method, max_depth, n_features, depth + 1)
   131:                 for _ in range(arity)]
   132:     return Node(op_name, children)
   133: 
   134: 
   135: def ramped_half_and_half(pop_size, max_depth, n_features):
   136:     """Initialize population with ramped half-and-half method."""
   137:     population = []
   138:     for i in range(pop_size):
   139:         depth = 2 + (i % (max_depth - 1))
   140:         method = 'full' if i % 2 == 0 else 'grow'
   141:         population.append(generate_tree(method, depth, n_features))
   142:     return population
   143: 
   144: 
   145: # ============================================================
   146: # Benchmark Data (FIXED)
   147: # ============================================================
   148: 
   149: BENCHMARKS = {
   150:     'nguyen7': {
   151:         'func': lambda X: np.log(X[:, 0] + 1) + np.log(X[:, 0] ** 2 + 1),
   152:         'n_features': 1,
   153:         'train_range': (0.0, 2.0),
   154:         'n_train': 20,
   155:         'test_range': (-0.5, 2.5),
   156:         'n_test': 100,
   157:     },
   158:     'nguyen10': {
   159:         'func': lambda X: 2 * np.sin(X[:, 0]) * np.cos(X[:, 1]),
   160:         'n_features': 2,
   161:         'train_range': (0.0, 2 * np.pi),
   162:         'n_train': 100,
   163:         'test_range': (0.0, 2 * np.pi),
   164:         'n_test': 400,
   165:     },
   166:     'koza3': {
   167:         'func': lambda X: X[:, 0] ** 5 - 2 * X[:, 0] ** 3 + X[:, 0],
   168:         'n_features': 1,
   169:         'train_range': (-1.0, 1.0),
   170:         'n_train': 20,
   171:         'test_range': (-1.0, 1.0),
   172:         'n_test': 100,
   173:     },
   174: }
   175: 
   176: 
   177: def generate_data(benchmark_name, seed=42):
   178:     """Generate train/test data for a benchmark function."""
   179:     bench = BENCHMARKS[benchmark_name]
   180:     n_features = bench['n_features']
   181:     lo, hi = bench['train_range']
   182: 
   183:     if n_features == 1:
   184:         X_train = np.linspace(lo, hi, bench['n_train']).reshape(-1, 1)
   185:         lo_t, hi_t = bench['test_range']
   186:         X_test = np.linspace(lo_t, hi_t, bench['n_test']).reshape(-1, 1)
   187:     else:
   188:         n_per_dim = int(round(bench['n_train'] ** (1.0 / n_features)))
   189:         grids = [np.linspace(lo, hi, n_per_dim) for _ in range(n_features)]
   190:         X_train = np.array(np.meshgrid(*grids)).T.reshape(-1, n_features)
   191:         lo_t, hi_t = bench['test_range']
   192:         n_per_dim_t = int(round(bench['n_test'] ** (1.0 / n_features)))
   193:         grids_t = [np.linspace(lo_t, hi_t, n_per_dim_t) for _ in range(n_features)]
   194:         X_test = np.array(np.meshgrid(*grids_t)).T.reshape(-1, n_features)
   195: 
   196:     y_train = bench['func'](X_train)
   197:     y_test = bench['func'](X_test)
   198:     return X_train, y_train, X_test, y_test, n_features
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
   214: 
   215: def r2_score(y_true, y_pred):
   216:     """Compute R-squared score."""
   217:     ss_res = np.sum((y_true - y_pred) ** 2)
   218:     ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
   219:     if ss_tot < 1e-15:
   220:         return 1.0 if ss_res < 1e-15 else 0.0
   221:     return max(1.0 - ss_res / ss_tot, 0.0)  # SRBench floor: clip blowups
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
   312: 
   313: def main():
   314:     parser = argparse.ArgumentParser(description="GP Symbolic Regression")
   315:     parser.add_argument('--benchmark', type=str, required=True,
   316:                         choices=list(BENCHMARKS.keys()))
   317:     parser.add_argument('--seed', type=int, default=42)
   318:     parser.add_argument('--pop-size', type=int, default=500)
   319:     parser.add_argument('--generations', type=int, default=50)
   320:     parser.add_argument('--max-depth', type=int, default=6)
   321:     args = parser.parse_args()
   322: 
   323:     random.seed(args.seed)
   324:     np.random.seed(args.seed)
   325: 
   326:     X_train, y_train, X_test, y_test, n_features = generate_data(
   327:         args.benchmark, args.seed
   328:     )
   329: 
   330:     # Initialize population
   331:     population = ramped_half_and_half(args.pop_size, args.max_depth, n_features)
   332: 
   333:     best_fitness_ever = float('inf')
   334:     best_tree_ever = None
   335: 
   336:     for gen in range(args.generations):
   337:         fitnesses = [fitness_function(tree, X_train, y_train)
   338:                      for tree in population]
   339: 
   340:         best_idx = int(np.argmin(fitnesses))
   341:         best_fitness = fitnesses[best_idx]
   342:         avg_fitness = float(np.mean(fitnesses))
   343:         best_size = population[best_idx].size()
   344: 
   345:         if best_fitness < best_fitness_ever:
   346:             best_fitness_ever = best_fitness
   347:             best_tree_ever = population[best_idx].copy()
   348: 
   349:         y_pred_gen = safe_evaluate(best_tree_ever, X_train)
   350:         train_r2 = r2_score(y_train, y_pred_gen)
   351: 
   352:         print(
   353:             f"TRAIN_METRICS generation={gen} best_fitness={best_fitness:.6f} "
   354:             f"avg_fitness={avg_fitness:.6f} best_size={best_size} "
   355:             f"train_r2={train_r2:.6f}",
   356:             flush=True,
   357:         )
   358: 
   359:         if gen < args.generations - 1:
   360:             population = evolve_one_generation(
   361:                 population, fitnesses, X_train, y_train,
   362:                 n_features, args.pop_size,
   363:                 max_depth=args.max_depth + 2,
   364:             )
   365: 
   366:     # Final evaluation on test set
   367:     y_pred_train = safe_evaluate(best_tree_ever, X_train)
   368:     y_pred_test = safe_evaluate(best_tree_ever, X_test)
   369:     train_r2 = r2_score(y_train, y_pred_train)
   370:     test_r2 = r2_score(y_test, y_pred_test)
   371:     test_rmse = float(np.sqrt(np.mean((y_test - y_pred_test) ** 2)))
   372:     expr_str = str(best_tree_ever)
   373: 
   374:     print(
   375:         f"TEST_METRICS r2={test_r2:.6f} rmse={test_rmse:.6f} "
   376:         f"train_r2={train_r2:.6f} size={best_tree_ever.size()} "
   377:         f'expression="{expr_str}"',
   378:         flush=True,
   379:     )
   380: 
   381: 
   382: if __name__ == '__main__':
   383:     main()
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `standard_gp` baseline — editable region  [READ-ONLY — reference implementation]

In `gplearn/custom_sr.py`:

```python
Lines 228–325:
   225: # Search Strategy (EDITABLE)
   226: # ============================================================
   227: 
   228: def fitness_function(tree, X, y):
   229:     """MSE fitness — lower is better."""
   230:     y_pred = safe_evaluate(tree, X)
   231:     return float(np.mean((y - y_pred) ** 2))
   232: 
   233: 
   234: def selection(population, fitnesses, n_select, tournament_size=7):
   235:     """Tournament selection."""
   236:     selected = []
   237:     pop_size = len(population)
   238:     for _ in range(n_select):
   239:         candidates = random.sample(range(pop_size), min(tournament_size, pop_size))
   240:         best = min(candidates, key=lambda i: fitnesses[i])
   241:         selected.append(population[best].copy())
   242:     return selected
   243: 
   244: 
   245: def crossover(parent1, parent2, n_features, max_depth=17):
   246:     """Standard subtree crossover."""
   247:     offspring = parent1.copy()
   248:     donor = parent2.copy()
   249: 
   250:     # Pick random crossover points
   251:     off_size = offspring.size()
   252:     don_size = donor.size()
   253:     if off_size <= 1 or don_size <= 1:
   254:         return offspring
   255: 
   256:     off_point = random.randint(1, off_size - 1)
   257:     don_point = random.randint(0, don_size - 1)
   258: 
   259:     # Extract donor subtree
   260:     donor_nodes = donor.get_all_nodes()
   261:     donor_subtree = donor_nodes[don_point][0].copy()
   262: 
   263:     # Replace in offspring
   264:     off_nodes = offspring.get_all_nodes()
   265:     node, parent, child_idx = off_nodes[off_point]
   266:     if parent is not None:
   267:         parent.children[child_idx] = donor_subtree
   268:     else:
   269:         offspring = donor_subtree
   270: 
   271:     # Reject if too deep
   272:     if offspring.depth() > max_depth:
   273:         return parent1.copy()
   274: 
   275:     return offspring
   276: 
   277: 
   278: def mutation(parent, n_features, max_depth=17):
   279:     """Subtree mutation — replace a random subtree with a new random tree."""
   280:     offspring = parent.copy()
   281:     tree_size = offspring.size()
   282:     if tree_size <= 1:
   283:         return generate_tree('grow', 3, n_features)
   284: 
   285:     mut_point = random.randint(1, tree_size - 1)
   286:     new_subtree = generate_tree('grow', 3, n_features)
   287: 
   288:     nodes = offspring.get_all_nodes()
   289:     node, par, child_idx = nodes[mut_point]
   290:     if par is not None:
   291:         par.children[child_idx] = new_subtree
   292:     else:
   293:         offspring = new_subtree
   294: 
   295:     if offspring.depth() > max_depth:
   296:         return parent.copy()
   297: 
   298:     return offspring
   299: 
   300: 
   301: def evolve_one_generation(population, fitnesses, X_train, y_train,
   302:                           n_features, pop_size,
   303:                           crossover_rate=0.9, mutation_rate=0.05,
   304:                           max_depth=17):
   305:     """Standard GP generation with elitism."""
   306:     new_population = []
   307: 
   308:     # Elitism: keep best
   309:     elite_idx = int(np.argmin(fitnesses))
   310:     new_population.append(population[elite_idx].copy())
   311: 
   312:     while len(new_population) < pop_size:
   313:         r = random.random()
   314:         if r < crossover_rate:
   315:             parents = selection(population, fitnesses, 2)
   316:             child = crossover(parents[0], parents[1], n_features, max_depth)
   317:         elif r < crossover_rate + mutation_rate:
   318:             parents = selection(population, fitnesses, 1)
   319:             child = mutation(parents[0], n_features, max_depth)
   320:         else:
   321:             parents = selection(population, fitnesses, 1)
   322:             child = parents[0]
   323:         new_population.append(child)
   324: 
   325:     return new_population[:pop_size]
   326: 
   327: 
   328: # ============================================================
```

### `parsimony_gp` baseline — editable region  [READ-ONLY — reference implementation]

In `gplearn/custom_sr.py`:

```python
Lines 228–347:
   225: # Search Strategy (EDITABLE)
   226: # ============================================================
   227: 
   228: def fitness_function(tree, X, y):
   229:     """Raw MSE fitness — lower is better.
   230: 
   231:     Parsimony pressure is applied at the population level inside
   232:     evolve_one_generation, not here. This ensures best_tree_ever
   233:     in the main loop tracks the best-fitting tree by actual MSE.
   234:     """
   235:     y_pred = safe_evaluate(tree, X)
   236:     return float(np.mean((y - y_pred) ** 2))
   237: 
   238: 
   239: def selection(population, fitnesses, n_select, tournament_size=7):
   240:     """Tournament selection on (possibly penalized) fitnesses."""
   241:     selected = []
   242:     pop_size = len(population)
   243:     for _ in range(n_select):
   244:         candidates = random.sample(range(pop_size), min(tournament_size, pop_size))
   245:         best = min(candidates, key=lambda i: fitnesses[i])
   246:         selected.append(population[best].copy())
   247:     return selected
   248: 
   249: 
   250: def crossover(parent1, parent2, n_features, max_depth=17):
   251:     """Standard subtree crossover."""
   252:     offspring = parent1.copy()
   253:     donor = parent2.copy()
   254: 
   255:     off_size = offspring.size()
   256:     don_size = donor.size()
   257:     if off_size <= 1 or don_size <= 1:
   258:         return offspring
   259: 
   260:     off_point = random.randint(1, off_size - 1)
   261:     don_point = random.randint(0, don_size - 1)
   262: 
   263:     donor_nodes = donor.get_all_nodes()
   264:     donor_subtree = donor_nodes[don_point][0].copy()
   265: 
   266:     off_nodes = offspring.get_all_nodes()
   267:     node, parent, child_idx = off_nodes[off_point]
   268:     if parent is not None:
   269:         parent.children[child_idx] = donor_subtree
   270:     else:
   271:         offspring = donor_subtree
   272: 
   273:     if offspring.depth() > max_depth:
   274:         return parent1.copy()
   275: 
   276:     return offspring
   277: 
   278: 
   279: def mutation(parent, n_features, max_depth=17):
   280:     """Subtree mutation — replace a random subtree with a new random tree."""
   281:     offspring = parent.copy()
   282:     tree_size = offspring.size()
   283:     if tree_size <= 1:
   284:         return generate_tree('grow', 3, n_features)
   285: 
   286:     mut_point = random.randint(1, tree_size - 1)
   287:     new_subtree = generate_tree('grow', 3, n_features)
   288: 
   289:     nodes = offspring.get_all_nodes()
   290:     node, par, child_idx = nodes[mut_point]
   291:     if par is not None:
   292:         par.children[child_idx] = new_subtree
   293:     else:
   294:         offspring = new_subtree
   295: 
   296:     if offspring.depth() > max_depth:
   297:         return parent.copy()
   298: 
   299:     return offspring
   300: 
   301: 
   302: def evolve_one_generation(population, fitnesses, X_train, y_train,
   303:                           n_features, pop_size,
   304:                           crossover_rate=0.9, mutation_rate=0.05,
   305:                           max_depth=17):
   306:     """Parsimony GP generation with parsimony pressure for bloat control.
   307: 
   308:     Uses gplearn-style auto parsimony coefficient computed per generation:
   309:         c = Cov(length, fitness) / Var(length)
   310:     clamped to [0, 0.001] to prevent runaway penalization.
   311:     Parsimony pressure is applied only during selection; elitism uses
   312:     raw fitness so the best-fitting individual is always preserved.
   313:     """
   314:     new_population = []
   315: 
   316:     # Adaptive parsimony coefficient (gplearn 'auto' method, clamped)
   317:     lengths = np.array([tree.size() for tree in population], dtype=float)
   318:     raw_fit = np.array(fitnesses, dtype=float)
   319:     len_var = float(np.var(lengths))
   320:     if len_var > 1e-15:
   321:         parsimony_coeff = float(np.cov(lengths, raw_fit)[1, 0]) / len_var
   322:         parsimony_coeff = max(parsimony_coeff, 0.0)
   323:         parsimony_coeff = min(parsimony_coeff, 0.001)
   324:     else:
   325:         parsimony_coeff = 0.0
   326: 
   327:     # Penalized fitnesses for selection only
   328:     penalized = [f + parsimony_coeff * l for f, l in zip(fitnesses, lengths)]
   329: 
   330:     # Elitism: keep best by raw fitness (not penalized)
   331:     elite_idx = int(np.argmin(fitnesses))
   332:     new_population.append(population[elite_idx].copy())
   333: 
   334:     while len(new_population) < pop_size:
   335:         r = random.random()
   336:         if r < crossover_rate:
   337:             parents = selection(population, penalized, 2)
   338:             child = crossover(parents[0], parents[1], n_features, max_depth)
   339:         elif r < crossover_rate + mutation_rate:
   340:             parents = selection(population, penalized, 1)
   341:             child = mutation(parents[0], n_features, max_depth)
   342:         else:
   343:             parents = selection(population, penalized, 1)
   344:             child = parents[0]
   345:         new_population.append(child)
   346: 
   347:     return new_population[:pop_size]
   348: 
   349: 
   350: # ============================================================
```

### `lexicase_gp` baseline — editable region  [READ-ONLY — reference implementation]

In `gplearn/custom_sr.py`:

```python
Lines 228–367:
   225: # Search Strategy (EDITABLE)
   226: # ============================================================
   227: 
   228: def fitness_function(tree, X, y):
   229:     """MSE fitness — lower is better."""
   230:     y_pred = safe_evaluate(tree, X)
   231:     return float(np.mean((y - y_pred) ** 2))
   232: 
   233: 
   234: def _per_case_errors(population, X, y):
   235:     """Compute per-case absolute errors for the entire population.
   236: 
   237:     Returns:
   238:         numpy array of shape (len(population), n_samples)
   239:     """
   240:     errors = np.empty((len(population), X.shape[0]))
   241:     for i, tree in enumerate(population):
   242:         y_pred = safe_evaluate(tree, X)
   243:         errors[i] = np.abs(y - y_pred)
   244:     return errors
   245: 
   246: 
   247: def selection(population, fitnesses, n_select, _errors=None, _X=None, _y=None):
   248:     """Epsilon-lexicase selection.
   249: 
   250:     Requires _errors (per-case errors), _X, _y to be passed via
   251:     evolve_one_generation. Falls back to tournament if not available.
   252:     """
   253:     selected = []
   254:     pop_size = len(population)
   255: 
   256:     if _errors is None:
   257:         # Fallback to tournament
   258:         for _ in range(n_select):
   259:             candidates = random.sample(range(pop_size), min(7, pop_size))
   260:             best = min(candidates, key=lambda i: fitnesses[i])
   261:             selected.append(population[best].copy())
   262:         return selected
   263: 
   264:     n_cases = _errors.shape[1]
   265:     for _ in range(n_select):
   266:         candidates = list(range(pop_size))
   267:         cases = list(range(n_cases))
   268:         random.shuffle(cases)
   269: 
   270:         for case in cases:
   271:             if len(candidates) <= 1:
   272:                 break
   273:             case_errors = _errors[candidates, case]
   274:             # Semi-dynamic epsilon-lexicase (La Cava 2016/2019): candidates
   275:             # survive iff their error ≤ best_on_case + MAD. The previous
   276:             # `median + MAD` admitted most of the population and degraded
   277:             # lexicase toward random selection.
   278:             min_err = float(np.min(case_errors))
   279:             mad = float(np.median(np.abs(case_errors - float(np.median(case_errors)))))
   280:             candidates = [c for c, e in zip(candidates, case_errors) if e <= min_err + mad]
   281: 
   282:         winner = random.choice(candidates)
   283:         selected.append(population[winner].copy())
   284: 
   285:     return selected
   286: 
   287: 
   288: def crossover(parent1, parent2, n_features, max_depth=17):
   289:     """Standard subtree crossover."""
   290:     offspring = parent1.copy()
   291:     donor = parent2.copy()
   292: 
   293:     off_size = offspring.size()
   294:     don_size = donor.size()
   295:     if off_size <= 1 or don_size <= 1:
   296:         return offspring
   297: 
   298:     off_point = random.randint(1, off_size - 1)
   299:     don_point = random.randint(0, don_size - 1)
   300: 
   301:     donor_nodes = donor.get_all_nodes()
   302:     donor_subtree = donor_nodes[don_point][0].copy()
   303: 
   304:     off_nodes = offspring.get_all_nodes()
   305:     node, parent, child_idx = off_nodes[off_point]
   306:     if parent is not None:
   307:         parent.children[child_idx] = donor_subtree
   308:     else:
   309:         offspring = donor_subtree
   310: 
   311:     if offspring.depth() > max_depth:
   312:         return parent1.copy()
   313: 
   314:     return offspring
   315: 
   316: 
   317: def mutation(parent, n_features, max_depth=17):
   318:     """Subtree mutation — replace a random subtree with a new random tree."""
   319:     offspring = parent.copy()
   320:     tree_size = offspring.size()
   321:     if tree_size <= 1:
   322:         return generate_tree('grow', 3, n_features)
   323: 
   324:     mut_point = random.randint(1, tree_size - 1)
   325:     new_subtree = generate_tree('grow', 3, n_features)
   326: 
   327:     nodes = offspring.get_all_nodes()
   328:     node, par, child_idx = nodes[mut_point]
   329:     if par is not None:
   330:         par.children[child_idx] = new_subtree
   331:     else:
   332:         offspring = new_subtree
   333: 
   334:     if offspring.depth() > max_depth:
   335:         return parent.copy()
   336: 
   337:     return offspring
   338: 
   339: 
   340: def evolve_one_generation(population, fitnesses, X_train, y_train,
   341:                           n_features, pop_size,
   342:                           crossover_rate=0.9, mutation_rate=0.05,
   343:                           max_depth=17):
   344:     """Lexicase GP generation — uses epsilon-lexicase selection."""
   345:     new_population = []
   346: 
   347:     # Elitism: keep best
   348:     elite_idx = int(np.argmin(fitnesses))
   349:     new_population.append(population[elite_idx].copy())
   350: 
   351:     # Pre-compute per-case errors for lexicase selection
   352:     errors = _per_case_errors(population, X_train, y_train)
   353: 
   354:     while len(new_population) < pop_size:
   355:         r = random.random()
   356:         if r < crossover_rate:
   357:             parents = selection(population, fitnesses, 2, _errors=errors)
   358:             child = crossover(parents[0], parents[1], n_features, max_depth)
   359:         elif r < crossover_rate + mutation_rate:
   360:             parents = selection(population, fitnesses, 1, _errors=errors)
   361:             child = mutation(parents[0], n_features, max_depth)
   362:         else:
   363:             parents = selection(population, fitnesses, 1, _errors=errors)
   364:             child = parents[0]
   365:         new_population.append(child)
   366: 
   367:     return new_population[:pop_size]
   368: 
   369: 
   370: # ============================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
