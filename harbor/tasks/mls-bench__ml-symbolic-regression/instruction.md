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
(inclusive, 1-indexed)**. Edits that change code outside these ranges — or creating new files, or
deleting whole files — will cause your submission to be invalid.

The line numbers mark an editable **region**, not a fixed line-count budget: you
may add or remove lines inside it. Only code outside the editable ranges must
stay unchanged.

- `gplearn/custom_sr.py`
- editable lines **188–266**


Other files you may **read** for context (do not modify):
- `gplearn/gplearn/genetic.py`
- `gplearn/gplearn/_program.py`
- `gplearn/gplearn/fitness.py`


## Readable Context


### `gplearn/custom_sr.py`  [EDITABLE — lines 188–266 only]

```python
     1: #!/usr/bin/env python3
     2: """Symbolic Regression via Genetic Programming.
     3: 
     4: A self-contained GP framework for symbolic regression. The editable section
     5: contains the search strategy: fitness function, selection, crossover, mutation,
     6: and per-generation evolution logic.
     7: 
     8: The benchmark identity (which target function is used) and the held-out test
     9: labels are NOT available here. The FIXED runner loads a pre-generated
    10: ``(X_train, y_train, X_test)`` triple — only SAMPLES of the target on the
    11: training inputs — drives the GP search using those samples for fitness, then
    12: emits the best evolved expression's predictions on ``X_test``. The host-side
    13: scorer regenerates the test labels and computes R2. Your GP must fit the
    14: ``(X_train, y_train)`` samples; there is no closed-form target to read off.
    15: """
    16: 
    17: import argparse
    18: import base64
    19: import io
    20: import math
    21: import os
    22: import random
    23: import sys
    24: 
    25: import numpy as np
    26: 
    27: 
    28: # ============================================================
    29: # Operator Definitions (FIXED)
    30: # ============================================================
    31: 
    32: def protected_div(a, b):
    33:     """Protected division: returns 1.0 when divisor is near zero."""
    34:     return np.where(np.abs(b) > 1e-10, a / b, 1.0)
    35: 
    36: 
    37: def protected_log(a):
    38:     """Protected log: returns 0.0 for non-positive inputs."""
    39:     return np.where(np.abs(a) > 1e-10, np.log(np.abs(a)), 0.0)
    40: 
    41: 
    42: def protected_exp(a):
    43:     """Protected exp: clips input to prevent overflow."""
    44:     return np.exp(np.clip(a, -10, 10))
    45: 
    46: 
    47: OPERATORS = {
    48:     'add': (np.add, 2),
    49:     'sub': (np.subtract, 2),
    50:     'mul': (np.multiply, 2),
    51:     'div': (protected_div, 2),
    52:     'sin': (np.sin, 1),
    53:     'cos': (np.cos, 1),
    54:     'log': (protected_log, 1),
    55:     'exp': (protected_exp, 1),
    56: }
    57: 
    58: OPERATOR_NAMES = list(OPERATORS.keys())
    59: 
    60: 
    61: # ============================================================
    62: # Tree Representation (FIXED)
    63: # ============================================================
    64: 
    65: class Node:
    66:     """A node in the GP expression tree."""
    67:     __slots__ = ('value', 'children')
    68: 
    69:     def __init__(self, value, children=None):
    70:         self.value = value
    71:         self.children = children or []
    72: 
    73:     @property
    74:     def is_terminal(self):
    75:         return len(self.children) == 0
    76: 
    77:     def evaluate(self, X):
    78:         """Evaluate expression tree on input array X (n_samples, n_features)."""
    79:         if self.is_terminal:
    80:             if isinstance(self.value, str) and self.value.startswith('x'):
    81:                 idx = int(self.value[1:])
    82:                 return X[:, idx].copy()
    83:             else:
    84:                 return np.full(X.shape[0], float(self.value))
    85:         func, arity = OPERATORS[self.value]
    86:         args = [child.evaluate(X) for child in self.children]
    87:         result = func(*args)
    88:         return np.clip(result, -1e15, 1e15)
    89: 
    90:     def size(self):
    91:         """Count total nodes in the tree."""
    92:         return 1 + sum(c.size() for c in self.children)
    93: 
    94:     def depth(self):
    95:         """Compute tree depth."""
    96:         if not self.children:
    97:             return 0
    98:         return 1 + max(c.depth() for c in self.children)
    99: 
   100:     def copy(self):
   101:         """Deep copy the tree."""
   102:         return Node(self.value, [c.copy() for c in self.children])
   103: 
   104:     def get_all_nodes(self):
   105:         """Return a list of (node, parent, child_index) via preorder traversal."""
   106:         result = [(self, None, None)]
   107:         for i, child in enumerate(self.children):
   108:             child_nodes = child.get_all_nodes()
   109:             # Update parent info for direct children
   110:             child_nodes[0] = (child, self, i)
   111:             result.extend(child_nodes)
   112:         return result
   113: 
   114:     def __str__(self):
   115:         if self.is_terminal:
   116:             return str(self.value)
   117:         if len(self.children) == 1:
   118:             return f"{self.value}({self.children[0]})"
   119:         return f"({self.children[0]} {self.value} {self.children[1]})"
   120: 
   121: 
   122: # ============================================================
   123: # Tree Generation (FIXED)
   124: # ============================================================
   125: 
   126: def random_terminal(n_features, const_range=(-5.0, 5.0)):
   127:     """Generate a random terminal node (variable or constant)."""
   128:     if random.random() < 0.5:
   129:         idx = random.randint(0, n_features - 1)
   130:         return Node(f'x{idx}')
   131:     else:
   132:         return Node(str(round(random.uniform(*const_range), 2)))
   133: 
   134: 
   135: def generate_tree(method, max_depth, n_features, depth=0):
   136:     """Generate a random expression tree using 'grow' or 'full' method."""
   137:     if depth >= max_depth or (method == 'grow' and depth > 0 and random.random() < 0.3):
   138:         return random_terminal(n_features)
   139:     op_name = random.choice(OPERATOR_NAMES)
   140:     _, arity = OPERATORS[op_name]
   141:     children = [generate_tree(method, max_depth, n_features, depth + 1)
   142:                 for _ in range(arity)]
   143:     return Node(op_name, children)
   144: 
   145: 
   146: def ramped_half_and_half(pop_size, max_depth, n_features):
   147:     """Initialize population with ramped half-and-half method."""
   148:     population = []
   149:     for i in range(pop_size):
   150:         depth = 2 + (i % (max_depth - 1))
   151:         method = 'full' if i % 2 == 0 else 'grow'
   152:         population.append(generate_tree(method, depth, n_features))
   153:     return population
   154: 
   155: 
   156: # ============================================================
   157: # Evaluation Utilities (FIXED)
   158: # ============================================================
   159: 
   160: def safe_evaluate(tree, X):
   161:     """Evaluate tree with error handling."""
   162:     try:
   163:         result = tree.evaluate(X)
   164:         result = np.nan_to_num(result, nan=1e10, posinf=1e10, neginf=-1e10)
   165:         return np.clip(result, -1e10, 1e10)
   166:     except Exception:
   167:         return np.full(X.shape[0], 1e10)
   168: 
   169: 
   170: def _train_r2(y_true, y_pred):
   171:     """R2 of the GP fit on the TRAINING samples (feedback only).
   172: 
   173:     This uses only the (X_train, y_train) samples the search already has access
   174:     to, so it leaks nothing about the held-out test target. The official test
   175:     R2 is computed host-side from the emitted predictions.
   176:     """
   177:     ss_res = np.sum((y_true - y_pred) ** 2)
   178:     ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
   179:     if ss_tot < 1e-15:
   180:         return 1.0 if ss_res < 1e-15 else 0.0
   181:     return max(1.0 - ss_res / ss_tot, 0.0)
   182: 
   183: 
   184: # ============================================================
   185: # Search Strategy (EDITABLE)
   186: # ============================================================
   187: 
   188: def fitness_function(tree, X, y):
   189:     """Evaluate fitness of a candidate program. Lower is better."""
   190:     y_pred = safe_evaluate(tree, X)
   191:     return float(np.mean((y - y_pred) ** 2))
   192: 
   193: 
   194: def selection(population, fitnesses, n_select):
   195:     """Select individuals from the population for reproduction.
   196: 
   197:     Args:
   198:         population: list of Node trees
   199:         fitnesses: list of float fitness values (lower is better)
   200:         n_select: int number of individuals to select
   201: 
   202:     Returns:
   203:         list of Node copies of selected individuals
   204:     """
   205:     selected = []
   206:     for _ in range(n_select):
   207:         idx = random.randint(0, len(population) - 1)
   208:         selected.append(population[idx].copy())
   209:     return selected
   210: 
   211: 
   212: def crossover(parent1, parent2, n_features, max_depth=17):
   213:     """Perform crossover between two parent trees.
   214: 
   215:     Returns:
   216:         Node - offspring tree
   217:     """
   218:     return parent1.copy()
   219: 
   220: 
   221: def mutation(parent, n_features, max_depth=17):
   222:     """Perform mutation on a parent tree.
   223: 
   224:     Returns:
   225:         Node - mutated tree
   226:     """
   227:     return parent.copy()
   228: 
   229: 
   230: def evolve_one_generation(population, fitnesses, X_train, y_train,
   231:                           n_features, pop_size,
   232:                           crossover_rate=0.9, mutation_rate=0.05,
   233:                           max_depth=17):
   234:     """Create the next generation from the current population.
   235: 
   236:     Args:
   237:         population: list of Node trees
   238:         fitnesses: list of float fitness values (lower is better)
   239:         X_train: numpy array (n_samples, n_features) - training inputs
   240:         y_train: numpy array (n_samples,) - training targets
   241:         n_features: number of input features
   242:         pop_size: desired population size
   243:         crossover_rate: probability of crossover
   244:         mutation_rate: probability of mutation
   245:         max_depth: maximum allowed tree depth
   246: 
   247:     Returns:
   248:         list of Node - next generation population
   249:     """
   250:     new_population = []
   251:     # Elitism: keep best individual
   252:     elite_idx = int(np.argmin(fitnesses))
   253:     new_population.append(population[elite_idx].copy())
   254: 
   255:     while len(new_population) < pop_size:
   256:         parents = selection(population, fitnesses, 2)
   257:         r = random.random()
   258:         if r < crossover_rate:
   259:             child = crossover(parents[0], parents[1], n_features, max_depth)
   260:         elif r < crossover_rate + mutation_rate:
   261:             child = mutation(parents[0], n_features, max_depth)
   262:         else:
   263:             child = parents[0]
   264:         new_population.append(child)
   265: 
   266:     return new_population[:pop_size]
   267: 
   268: 
   269: # ============================================================
   270: # FIXED: input loading + GP driver + prediction emit
   271: # (do not modify below this line)
   272: # ============================================================
   273: 
   274: def _inputs_dir():
   275:     d = os.environ.get("SR_INPUTS_DIR")
   276:     if d:
   277:         return d
   278:     return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_sr_inputs")
   279: 
   280: 
   281: def _load_input(task_id, seed):
   282:     """Load the pre-generated (X_train, y_train, X_test) for this run.
   283: 
   284:     Only training SAMPLES (X_train, y_train) and the test inputs (X_test) are
   285:     present; the closed-form target and the test labels are withheld.
   286:     """
   287:     path = os.path.join(_inputs_dir(), f"{task_id}_seed{seed}.npz.b64")
   288:     with open(path, "r") as f:
   289:         raw = base64.b64decode(f.read())
   290:     d = np.load(io.BytesIO(raw))
   291:     return d["X_train"], d["y_train"], d["X_test"]
   292: 
   293: 
   294: def main():
   295:     parser = argparse.ArgumentParser(description="GP Symbolic Regression")
   296:     parser.add_argument('--seed', type=int, default=42)
   297:     parser.add_argument('--pop-size', type=int, default=500)
   298:     parser.add_argument('--generations', type=int, default=50)
   299:     parser.add_argument('--max-depth', type=int, default=6)
   300:     args = parser.parse_args()
   301: 
   302:     # Opaque task id used only to locate the pre-generated inputs; it carries
   303:     # no information about which target function is in use.
   304:     task_id = os.environ.get("SR_TASK", "")
   305:     if not task_id:
   306:         raise SystemExit("SR_TASK not set")
   307: 
   308:     random.seed(args.seed)
   309:     np.random.seed(args.seed)
   310: 
   311:     X_train, y_train, X_test = _load_input(task_id, args.seed)
   312:     n_features = X_train.shape[1]
   313: 
   314:     # Initialize population
   315:     population = ramped_half_and_half(args.pop_size, args.max_depth, n_features)
   316: 
   317:     best_fitness_ever = float('inf')
   318:     best_tree_ever = None
   319: 
   320:     for gen in range(args.generations):
   321:         fitnesses = [fitness_function(tree, X_train, y_train)
   322:                      for tree in population]
   323: 
   324:         best_idx = int(np.argmin(fitnesses))
   325:         best_fitness = fitnesses[best_idx]
   326:         avg_fitness = float(np.mean(fitnesses))
   327:         best_size = population[best_idx].size()
   328: 
   329:         if best_fitness < best_fitness_ever:
   330:             best_fitness_ever = best_fitness
   331:             best_tree_ever = population[best_idx].copy()
   332: 
   333:         y_pred_gen = safe_evaluate(best_tree_ever, X_train)
   334:         train_r2 = _train_r2(y_train, y_pred_gen)
   335: 
   336:         print(
   337:             f"TRAIN_METRICS generation={gen} best_fitness={best_fitness:.6f} "
   338:             f"avg_fitness={avg_fitness:.6f} best_size={best_size} "
   339:             f"train_r2={train_r2:.6f}",
   340:             flush=True,
   341:         )
   342: 
   343:         if gen < args.generations - 1:
   344:             population = evolve_one_generation(
   345:                 population, fitnesses, X_train, y_train,
   346:                 n_features, args.pop_size,
   347:                 max_depth=args.max_depth + 2,
   348:             )
   349: 
   350:     # Final fit summary on the training samples (feedback only)
   351:     y_pred_train = safe_evaluate(best_tree_ever, X_train)
   352:     train_r2 = _train_r2(y_train, y_pred_train)
   353:     expr_str = str(best_tree_ever)
   354: 
   355:     # Emit predictions on the held-out test inputs for host-side scoring.
   356:     y_pred_test = safe_evaluate(best_tree_ever, X_test)
   357:     y_pred_test = np.ascontiguousarray(np.asarray(y_pred_test, dtype=np.float64)).ravel()
   358:     payload = base64.b64encode(y_pred_test.tobytes()).decode("ascii")
   359: 
   360:     print(
   361:         f"TEST_METRICS train_r2={train_r2:.6f} size={best_tree_ever.size()} "
   362:         f'expression="{expr_str}"',
   363:         flush=True,
   364:     )
   365:     print(
   366:         f"SR_PRED task={task_id} seed={args.seed} n={y_pred_test.shape[0]} "
   367:         f"preds={payload}",
   368:         flush=True,
   369:     )
   370: 
   371: 
   372: if __name__ == '__main__':
   373:     main()
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
Lines 188–285:
   185: # Search Strategy (EDITABLE)
   186: # ============================================================
   187: 
   188: def fitness_function(tree, X, y):
   189:     """MSE fitness — lower is better."""
   190:     y_pred = safe_evaluate(tree, X)
   191:     return float(np.mean((y - y_pred) ** 2))
   192: 
   193: 
   194: def selection(population, fitnesses, n_select, tournament_size=7):
   195:     """Tournament selection."""
   196:     selected = []
   197:     pop_size = len(population)
   198:     for _ in range(n_select):
   199:         candidates = random.sample(range(pop_size), min(tournament_size, pop_size))
   200:         best = min(candidates, key=lambda i: fitnesses[i])
   201:         selected.append(population[best].copy())
   202:     return selected
   203: 
   204: 
   205: def crossover(parent1, parent2, n_features, max_depth=17):
   206:     """Standard subtree crossover."""
   207:     offspring = parent1.copy()
   208:     donor = parent2.copy()
   209: 
   210:     # Pick random crossover points
   211:     off_size = offspring.size()
   212:     don_size = donor.size()
   213:     if off_size <= 1 or don_size <= 1:
   214:         return offspring
   215: 
   216:     off_point = random.randint(1, off_size - 1)
   217:     don_point = random.randint(0, don_size - 1)
   218: 
   219:     # Extract donor subtree
   220:     donor_nodes = donor.get_all_nodes()
   221:     donor_subtree = donor_nodes[don_point][0].copy()
   222: 
   223:     # Replace in offspring
   224:     off_nodes = offspring.get_all_nodes()
   225:     node, parent, child_idx = off_nodes[off_point]
   226:     if parent is not None:
   227:         parent.children[child_idx] = donor_subtree
   228:     else:
   229:         offspring = donor_subtree
   230: 
   231:     # Reject if too deep
   232:     if offspring.depth() > max_depth:
   233:         return parent1.copy()
   234: 
   235:     return offspring
   236: 
   237: 
   238: def mutation(parent, n_features, max_depth=17):
   239:     """Subtree mutation — replace a random subtree with a new random tree."""
   240:     offspring = parent.copy()
   241:     tree_size = offspring.size()
   242:     if tree_size <= 1:
   243:         return generate_tree('grow', 3, n_features)
   244: 
   245:     mut_point = random.randint(1, tree_size - 1)
   246:     new_subtree = generate_tree('grow', 3, n_features)
   247: 
   248:     nodes = offspring.get_all_nodes()
   249:     node, par, child_idx = nodes[mut_point]
   250:     if par is not None:
   251:         par.children[child_idx] = new_subtree
   252:     else:
   253:         offspring = new_subtree
   254: 
   255:     if offspring.depth() > max_depth:
   256:         return parent.copy()
   257: 
   258:     return offspring
   259: 
   260: 
   261: def evolve_one_generation(population, fitnesses, X_train, y_train,
   262:                           n_features, pop_size,
   263:                           crossover_rate=0.9, mutation_rate=0.05,
   264:                           max_depth=17):
   265:     """Standard GP generation with elitism."""
   266:     new_population = []
   267: 
   268:     # Elitism: keep best
   269:     elite_idx = int(np.argmin(fitnesses))
   270:     new_population.append(population[elite_idx].copy())
   271: 
   272:     while len(new_population) < pop_size:
   273:         r = random.random()
   274:         if r < crossover_rate:
   275:             parents = selection(population, fitnesses, 2)
   276:             child = crossover(parents[0], parents[1], n_features, max_depth)
   277:         elif r < crossover_rate + mutation_rate:
   278:             parents = selection(population, fitnesses, 1)
   279:             child = mutation(parents[0], n_features, max_depth)
   280:         else:
   281:             parents = selection(population, fitnesses, 1)
   282:             child = parents[0]
   283:         new_population.append(child)
   284: 
   285:     return new_population[:pop_size]
   286: 
   287: 
   288: # ============================================================
```

### `parsimony_gp` baseline — editable region  [READ-ONLY — reference implementation]

In `gplearn/custom_sr.py`:

```python
Lines 188–307:
   185: # Search Strategy (EDITABLE)
   186: # ============================================================
   187: 
   188: def fitness_function(tree, X, y):
   189:     """Raw MSE fitness — lower is better.
   190: 
   191:     Parsimony pressure is applied at the population level inside
   192:     evolve_one_generation, not here. This ensures best_tree_ever
   193:     in the main loop tracks the best-fitting tree by actual MSE.
   194:     """
   195:     y_pred = safe_evaluate(tree, X)
   196:     return float(np.mean((y - y_pred) ** 2))
   197: 
   198: 
   199: def selection(population, fitnesses, n_select, tournament_size=7):
   200:     """Tournament selection on (possibly penalized) fitnesses."""
   201:     selected = []
   202:     pop_size = len(population)
   203:     for _ in range(n_select):
   204:         candidates = random.sample(range(pop_size), min(tournament_size, pop_size))
   205:         best = min(candidates, key=lambda i: fitnesses[i])
   206:         selected.append(population[best].copy())
   207:     return selected
   208: 
   209: 
   210: def crossover(parent1, parent2, n_features, max_depth=17):
   211:     """Standard subtree crossover."""
   212:     offspring = parent1.copy()
   213:     donor = parent2.copy()
   214: 
   215:     off_size = offspring.size()
   216:     don_size = donor.size()
   217:     if off_size <= 1 or don_size <= 1:
   218:         return offspring
   219: 
   220:     off_point = random.randint(1, off_size - 1)
   221:     don_point = random.randint(0, don_size - 1)
   222: 
   223:     donor_nodes = donor.get_all_nodes()
   224:     donor_subtree = donor_nodes[don_point][0].copy()
   225: 
   226:     off_nodes = offspring.get_all_nodes()
   227:     node, parent, child_idx = off_nodes[off_point]
   228:     if parent is not None:
   229:         parent.children[child_idx] = donor_subtree
   230:     else:
   231:         offspring = donor_subtree
   232: 
   233:     if offspring.depth() > max_depth:
   234:         return parent1.copy()
   235: 
   236:     return offspring
   237: 
   238: 
   239: def mutation(parent, n_features, max_depth=17):
   240:     """Subtree mutation — replace a random subtree with a new random tree."""
   241:     offspring = parent.copy()
   242:     tree_size = offspring.size()
   243:     if tree_size <= 1:
   244:         return generate_tree('grow', 3, n_features)
   245: 
   246:     mut_point = random.randint(1, tree_size - 1)
   247:     new_subtree = generate_tree('grow', 3, n_features)
   248: 
   249:     nodes = offspring.get_all_nodes()
   250:     node, par, child_idx = nodes[mut_point]
   251:     if par is not None:
   252:         par.children[child_idx] = new_subtree
   253:     else:
   254:         offspring = new_subtree
   255: 
   256:     if offspring.depth() > max_depth:
   257:         return parent.copy()
   258: 
   259:     return offspring
   260: 
   261: 
   262: def evolve_one_generation(population, fitnesses, X_train, y_train,
   263:                           n_features, pop_size,
   264:                           crossover_rate=0.9, mutation_rate=0.05,
   265:                           max_depth=17):
   266:     """Parsimony GP generation with parsimony pressure for bloat control.
   267: 
   268:     Uses gplearn-style auto parsimony coefficient computed per generation:
   269:         c = Cov(length, fitness) / Var(length)
   270:     clamped to [0, 0.001] to prevent runaway penalization.
   271:     Parsimony pressure is applied only during selection; elitism uses
   272:     raw fitness so the best-fitting individual is always preserved.
   273:     """
   274:     new_population = []
   275: 
   276:     # Adaptive parsimony coefficient (gplearn 'auto' method, clamped)
   277:     lengths = np.array([tree.size() for tree in population], dtype=float)
   278:     raw_fit = np.array(fitnesses, dtype=float)
   279:     len_var = float(np.var(lengths))
   280:     if len_var > 1e-15:
   281:         parsimony_coeff = float(np.cov(lengths, raw_fit)[1, 0]) / len_var
   282:         parsimony_coeff = max(parsimony_coeff, 0.0)
   283:         parsimony_coeff = min(parsimony_coeff, 0.001)
   284:     else:
   285:         parsimony_coeff = 0.0
   286: 
   287:     # Penalized fitnesses for selection only
   288:     penalized = [f + parsimony_coeff * l for f, l in zip(fitnesses, lengths)]
   289: 
   290:     # Elitism: keep best by raw fitness (not penalized)
   291:     elite_idx = int(np.argmin(fitnesses))
   292:     new_population.append(population[elite_idx].copy())
   293: 
   294:     while len(new_population) < pop_size:
   295:         r = random.random()
   296:         if r < crossover_rate:
   297:             parents = selection(population, penalized, 2)
   298:             child = crossover(parents[0], parents[1], n_features, max_depth)
   299:         elif r < crossover_rate + mutation_rate:
   300:             parents = selection(population, penalized, 1)
   301:             child = mutation(parents[0], n_features, max_depth)
   302:         else:
   303:             parents = selection(population, penalized, 1)
   304:             child = parents[0]
   305:         new_population.append(child)
   306: 
   307:     return new_population[:pop_size]
   308: 
   309: 
   310: # ============================================================
```

### `lexicase_gp` baseline — editable region  [READ-ONLY — reference implementation]

In `gplearn/custom_sr.py`:

```python
Lines 188–327:
   185: # Search Strategy (EDITABLE)
   186: # ============================================================
   187: 
   188: def fitness_function(tree, X, y):
   189:     """MSE fitness — lower is better."""
   190:     y_pred = safe_evaluate(tree, X)
   191:     return float(np.mean((y - y_pred) ** 2))
   192: 
   193: 
   194: def _per_case_errors(population, X, y):
   195:     """Compute per-case absolute errors for the entire population.
   196: 
   197:     Returns:
   198:         numpy array of shape (len(population), n_samples)
   199:     """
   200:     errors = np.empty((len(population), X.shape[0]))
   201:     for i, tree in enumerate(population):
   202:         y_pred = safe_evaluate(tree, X)
   203:         errors[i] = np.abs(y - y_pred)
   204:     return errors
   205: 
   206: 
   207: def selection(population, fitnesses, n_select, _errors=None, _X=None, _y=None):
   208:     """Epsilon-lexicase selection.
   209: 
   210:     Requires _errors (per-case errors), _X, _y to be passed via
   211:     evolve_one_generation. Falls back to tournament if not available.
   212:     """
   213:     selected = []
   214:     pop_size = len(population)
   215: 
   216:     if _errors is None:
   217:         # Fallback to tournament
   218:         for _ in range(n_select):
   219:             candidates = random.sample(range(pop_size), min(7, pop_size))
   220:             best = min(candidates, key=lambda i: fitnesses[i])
   221:             selected.append(population[best].copy())
   222:         return selected
   223: 
   224:     n_cases = _errors.shape[1]
   225:     for _ in range(n_select):
   226:         candidates = list(range(pop_size))
   227:         cases = list(range(n_cases))
   228:         random.shuffle(cases)
   229: 
   230:         for case in cases:
   231:             if len(candidates) <= 1:
   232:                 break
   233:             case_errors = _errors[candidates, case]
   234:             # Semi-dynamic epsilon-lexicase (La Cava 2016/2019): candidates
   235:             # survive iff their error ≤ best_on_case + MAD. The previous
   236:             # `median + MAD` admitted most of the population and degraded
   237:             # lexicase toward random selection.
   238:             min_err = float(np.min(case_errors))
   239:             mad = float(np.median(np.abs(case_errors - float(np.median(case_errors)))))
   240:             candidates = [c for c, e in zip(candidates, case_errors) if e <= min_err + mad]
   241: 
   242:         winner = random.choice(candidates)
   243:         selected.append(population[winner].copy())
   244: 
   245:     return selected
   246: 
   247: 
   248: def crossover(parent1, parent2, n_features, max_depth=17):
   249:     """Standard subtree crossover."""
   250:     offspring = parent1.copy()
   251:     donor = parent2.copy()
   252: 
   253:     off_size = offspring.size()
   254:     don_size = donor.size()
   255:     if off_size <= 1 or don_size <= 1:
   256:         return offspring
   257: 
   258:     off_point = random.randint(1, off_size - 1)
   259:     don_point = random.randint(0, don_size - 1)
   260: 
   261:     donor_nodes = donor.get_all_nodes()
   262:     donor_subtree = donor_nodes[don_point][0].copy()
   263: 
   264:     off_nodes = offspring.get_all_nodes()
   265:     node, parent, child_idx = off_nodes[off_point]
   266:     if parent is not None:
   267:         parent.children[child_idx] = donor_subtree
   268:     else:
   269:         offspring = donor_subtree
   270: 
   271:     if offspring.depth() > max_depth:
   272:         return parent1.copy()
   273: 
   274:     return offspring
   275: 
   276: 
   277: def mutation(parent, n_features, max_depth=17):
   278:     """Subtree mutation — replace a random subtree with a new random tree."""
   279:     offspring = parent.copy()
   280:     tree_size = offspring.size()
   281:     if tree_size <= 1:
   282:         return generate_tree('grow', 3, n_features)
   283: 
   284:     mut_point = random.randint(1, tree_size - 1)
   285:     new_subtree = generate_tree('grow', 3, n_features)
   286: 
   287:     nodes = offspring.get_all_nodes()
   288:     node, par, child_idx = nodes[mut_point]
   289:     if par is not None:
   290:         par.children[child_idx] = new_subtree
   291:     else:
   292:         offspring = new_subtree
   293: 
   294:     if offspring.depth() > max_depth:
   295:         return parent.copy()
   296: 
   297:     return offspring
   298: 
   299: 
   300: def evolve_one_generation(population, fitnesses, X_train, y_train,
   301:                           n_features, pop_size,
   302:                           crossover_rate=0.9, mutation_rate=0.05,
   303:                           max_depth=17):
   304:     """Lexicase GP generation — uses epsilon-lexicase selection."""
   305:     new_population = []
   306: 
   307:     # Elitism: keep best
   308:     elite_idx = int(np.argmin(fitnesses))
   309:     new_population.append(population[elite_idx].copy())
   310: 
   311:     # Pre-compute per-case errors for lexicase selection
   312:     errors = _per_case_errors(population, X_train, y_train)
   313: 
   314:     while len(new_population) < pop_size:
   315:         r = random.random()
   316:         if r < crossover_rate:
   317:             parents = selection(population, fitnesses, 2, _errors=errors)
   318:             child = crossover(parents[0], parents[1], n_features, max_depth)
   319:         elif r < crossover_rate + mutation_rate:
   320:             parents = selection(population, fitnesses, 1, _errors=errors)
   321:             child = mutation(parents[0], n_features, max_depth)
   322:         else:
   323:             parents = selection(population, fitnesses, 1, _errors=errors)
   324:             child = parents[0]
   325:         new_population.append(child)
   326: 
   327:     return new_population[:pop_size]
   328: 
   329: 
   330: # ============================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
