# MLS-Bench: optimization-multi-objective

# Multi-Objective Optimization: Custom Evolutionary Strategy Design

## Research Question
Design a novel multi-objective evolutionary algorithm (MOEA) strategy that achieves better convergence, diversity, and spread on standard benchmark problems than classic approaches like NSGA-II, MOEA/D, and SPEA2.

## Background
Multi-objective optimization aims to find a set of Pareto-optimal solutions that represent the best trade-offs among conflicting objectives. Evolutionary algorithms are the dominant approach, differing primarily in three components:

- **Parent selection**: how to choose individuals for mating (e.g., tournament with crowding distance, reference-vector-based).
- **Variation**: how to produce offspring via crossover and mutation operators.
- **Environmental selection (survival)**: how to prune the combined parent + offspring pool back to population size (e.g., non-dominated sorting + crowding, decomposition into subproblems, indicator-based selection).

Classic algorithms:
- **NSGA-II** — non-dominated sorting + crowding distance for diversity (Deb, Pratap, Agarwal, and Meyarivan, *IEEE TEC* 6(2), 2002).
- **MOEA/D** — decomposes the problem into scalar subproblems via weight vectors (Zhang and Li, *IEEE TEC* 11(6), 2007).
- **SPEA2** — strength-based fitness with k-NN density estimation (Zitzler, Laumanns, and Thiele, EUROGEN 2001 / TIK-Report 103).

State-of-the-art:
- **NSGA-III** — reference-point-based selection for many-objective problems (Deb and Jain, *IEEE TEC* 18(4), 2014).
- **RVEA** — angle-penalized distance with adaptive reference vectors (Cheng, Jin, Olhofer, and Sendhoff, *IEEE TEC* 20(5), 2016).
- **AGE-MOEA** — adaptive geometry estimation for survival selection (Panichella, "An Adaptive Evolutionary Algorithm based on Non-Euclidean Geometry for Many-Objective Optimization", GECCO 2019).

## Task
Implement a custom multi-objective evolutionary strategy by modifying the `CustomMOEA` class in `deap/custom_moea.py`. You should implement the `select`, `vary`, `survive`, and optionally `on_generation` methods. The algorithm must work for both 2-objective and 3-objective problems.

## Interface
```python
class CustomMOEA:
    def __init__(self, pop_size, n_obj, n_var, bounds, cx_eta=20.0, mut_eta=20.0, mut_prob=None):
        """Initialize the MOEA with problem parameters."""

    def select(self, population: list, k: int) -> list:
        """Select k parents from the population for mating."""

    def vary(self, parents: list) -> list:
        """Apply crossover and mutation to produce offspring (fitness invalidated)."""

    def survive(self, population: list, offspring: list) -> list:
        """Environmental selection: choose pop_size individuals from combined pool."""

    def on_generation(self, gen: int, population: list):
        """Optional per-generation callback for adaptive strategies."""
```

Individual interface:
- `ind.fitness.values` -> tuple of objective values (all minimized).
- `ind.fitness.dominates(other.fitness)` -> bool.
- `ind.fitness.valid` -> bool (`True` if evaluated).

Available DEAP utilities:
- `tools.sortNondominated(pop, k)` -> list of fronts.
- `tools.selTournamentDCD(pop, k)` -> tournament selection (needs crowding distance).
- `tools.selNSGA3(pop, k, ref_points)` -> NSGA-III selection.
- `tools.cxSimulatedBinaryBounded(ind1, ind2, eta, low, up)` -> SBX crossover.
- `tools.mutPolynomialBounded(ind, eta, low, up, indpb)` -> polynomial mutation.
- `tools.uniform_reference_points(nobj, p)` -> generate reference points.
- `compute_crowding_distance(individuals)` -> sets `.fitness.crowding_dist`.
- `get_nondominated(population)` -> first non-dominated front.

## Baselines (paper-cited reference implementations)
- **nsga2** — Deb et al. (*IEEE TEC* 2002); paper-default SBX `eta_c = 20`, polynomial mutation `eta_m = 20`, `p_m = 1/n_var`.
- **moead** — Zhang and Li (*IEEE TEC* 2007); paper-default Tchebycheff aggregation, neighborhood size `T = 20`.
- **spea2** — Zitzler, Laumanns, and Thiele (EUROGEN 2001 / TIK-Report 103); paper-default archive size = population size, `k = sqrt(N + |archive|)` for k-NN density.
- **nsga3** — Deb and Jain (*IEEE TEC* 2014); paper-default Das–Dennis reference points with divisions chosen from objective dimensionality.
- **rvea** — Cheng, Jin, Olhofer, and Sendhoff (*IEEE TEC* 2016); paper-default angle-penalized distance with `alpha = 2`, reference-vector adaptation period `fr = 0.1`.
- **agemoea** — Panichella (GECCO 2019); paper-default geometry-estimated Minkowski-`p` survival.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/deap/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `deap/custom_moea.py`
- editable lines **297–441**




## Readable Context


### `deap/custom_moea.py`  [EDITABLE — lines 297–441 only]

```python
     1: """
     2: Multi-Objective Optimization — Custom Evolutionary Strategy Template
     3: 
     4: This script runs a complete multi-objective evolutionary algorithm on standard
     5: benchmark problems (ZDT/DTLZ). The agent should implement the custom selection
     6: and variation strategy in the CustomMOEA class.
     7: 
     8: Usage:
     9:     python deap/custom_moea.py --problem zdt1 --seed 42 --output-dir ./out
    10: """
    11: 
    12: import argparse
    13: import json
    14: import math
    15: import os
    16: import random
    17: import time
    18: import warnings
    19: from copy import deepcopy
    20: from functools import reduce
    21: from math import cos, pi, sin, sqrt
    22: from operator import mul
    23: from typing import List, Optional, Tuple
    24: 
    25: import numpy as np
    26: 
    27: from deap import base, benchmarks, creator, tools
    28: from deap.benchmarks import tools as btools
    29: 
    30: warnings.filterwarnings("ignore")
    31: 
    32: # ================================================================
    33: # FIXED — Problem definitions and utilities (do not modify)
    34: # ================================================================
    35: 
    36: # Create DEAP fitness and individual types
    37: creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    38: creator.create("Individual", list, fitness=creator.FitnessMin)
    39: 
    40: # For 3-objective problems
    41: creator.create("FitnessMin3", base.Fitness, weights=(-1.0, -1.0, -1.0))
    42: creator.create("Individual3", list, fitness=creator.FitnessMin3)
    43: 
    44: 
    45: PROBLEMS = {
    46:     "zdt1": {
    47:         "func": benchmarks.zdt1,
    48:         "n_var": 30,
    49:         "n_obj": 2,
    50:         "bounds": (0.0, 1.0),
    51:         "pop_size": 100,
    52:         "n_gen": 200,
    53:         "ref_point": [1.1, 1.1],
    54:         "description": "ZDT1: convex Pareto front, 30 variables, 2 objectives",
    55:     },
    56:     "zdt3": {
    57:         "func": benchmarks.zdt3,
    58:         "n_var": 30,
    59:         "n_obj": 2,
    60:         "bounds": (0.0, 1.0),
    61:         "pop_size": 100,
    62:         "n_gen": 200,
    63:         "ref_point": [1.1, 1.1],
    64:         "description": "ZDT3: disconnected Pareto front, 30 variables, 2 objectives",
    65:     },
    66:     "dtlz2": {
    67:         "func": lambda ind: benchmarks.dtlz2(ind, 3),
    68:         "n_var": 12,
    69:         "n_obj": 3,
    70:         "bounds": (0.0, 1.0),
    71:         "pop_size": 120,
    72:         "n_gen": 250,
    73:         "ref_point": [1.5, 1.5, 1.5],
    74:         "description": "DTLZ2: spherical Pareto front, 12 variables, 3 objectives",
    75:     },
    76:     "dtlz1": {
    77:         "func": lambda ind: benchmarks.dtlz1(ind, 3),
    78:         "n_var": 7,
    79:         "n_obj": 3,
    80:         "bounds": (0.0, 1.0),
    81:         "pop_size": 120,
    82:         "n_gen": 400,
    83:         "ref_point": [1.0, 1.0, 1.0],
    84:         "description": "DTLZ1: linear Pareto front with many local fronts, 7 variables, 3 objectives",
    85:     },
    86: }
    87: 
    88: 
    89: def generate_pareto_front(problem_name: str, n_points: int = 500) -> np.ndarray:
    90:     """Generate reference Pareto front points for IGD computation."""
    91:     if problem_name == "zdt1":
    92:         x = np.linspace(0, 1, n_points)
    93:         return np.column_stack([x, 1 - np.sqrt(x)])
    94:     elif problem_name == "zdt3":
    95:         # ZDT3 has a disconnected front
    96:         regions = [
    97:             (0.0, 0.0830),
    98:             (0.1822, 0.2577),
    99:             (0.4093, 0.4538),
   100:             (0.6183, 0.6525),
   101:             (0.8233, 0.8518),
   102:         ]
   103:         points = []
   104:         per_region = n_points // len(regions)
   105:         for lo, hi in regions:
   106:             x = np.linspace(lo, hi, per_region)
   107:             f1 = x
   108:             f2 = 1 - np.sqrt(x) - x * np.sin(10 * np.pi * x)
   109:             points.append(np.column_stack([f1, f2]))
   110:         return np.vstack(points)
   111:     elif problem_name == "dtlz2":
   112:         # Uniform points on first octant of unit sphere
   113:         points = []
   114:         ns = int(np.sqrt(n_points)) + 1
   115:         for i in range(ns):
   116:             for j in range(ns):
   117:                 theta1 = (i / max(ns - 1, 1)) * np.pi / 2
   118:                 theta2 = (j / max(ns - 1, 1)) * np.pi / 2
   119:                 f1 = np.cos(theta1) * np.cos(theta2)
   120:                 f2 = np.cos(theta1) * np.sin(theta2)
   121:                 f3 = np.sin(theta1)
   122:                 points.append([f1, f2, f3])
   123:         return np.array(points[:n_points])
   124:     elif problem_name == "dtlz1":
   125:         # Pareto front lies on the plane sum(f_i) = 0.5
   126:         points = []
   127:         ns = int(np.sqrt(n_points)) + 1
   128:         for i in range(ns):
   129:             for j in range(ns - i):
   130:                 f1 = i / max(ns - 1, 1) * 0.5
   131:                 f2 = j / max(ns - 1, 1) * 0.5
   132:                 f3 = 0.5 - f1 - f2
   133:                 if f3 >= -1e-8:
   134:                     points.append([f1, f2, max(f3, 0.0)])
   135:         return np.array(points[:n_points])
   136:     else:
   137:         raise ValueError(f"Unknown problem: {problem_name}")
   138: 
   139: 
   140: def _hv_2d(points, ref):
   141:     """Exact 2D hypervolume via non-dominated sweep."""
   142:     pts = points[(points[:, 0] < ref[0]) & (points[:, 1] < ref[1])]
   143:     if len(pts) == 0:
   144:         return 0.0
   145:     pts = pts[pts[:, 0].argsort()]
   146:     nd = [pts[0]]
   147:     for p in pts[1:]:
   148:         if p[1] < nd[-1][1]:
   149:             nd.append(p)
   150:     nd = np.array(nd)
   151:     hv = 0.0
   152:     prev_y = ref[1]
   153:     for p in nd:
   154:         width = ref[0] - p[0]
   155:         hv += width * (prev_y - p[1])
   156:         prev_y = p[1]
   157:     return hv
   158: 
   159: 
   160: def _hv_3d(points, ref):
   161:     """Exact 3D hypervolume via z-slicing + 2D sweep."""
   162:     mask = np.all(points < ref, axis=1)
   163:     pts = points[mask]
   164:     if len(pts) == 0:
   165:         return 0.0
   166:     # Sort by z ascending: as z increases, more points become active
   167:     order = np.argsort(pts[:, 2])
   168:     pts = pts[order]
   169:     hv = 0.0
   170:     active_2d = []
   171:     for i in range(len(pts)):
   172:         active_2d.append(pts[i, :2])
   173:         z_lo = pts[i, 2]
   174:         z_hi = pts[i + 1, 2] if i + 1 < len(pts) else ref[2]
   175:         dz = z_hi - z_lo
   176:         if dz > 0:
   177:             hv += _hv_2d(np.array(active_2d), ref[:2]) * dz
   178:     return hv
   179: 
   180: 
   181: def compute_hypervolume(nd_front, ref_point):
   182:     """Robust hypervolume computation that works for 2D and 3D.
   183: 
   184:     Falls back to a pure-Python implementation if DEAP's built-in fails.
   185:     """
   186:     # Always use pure-Python implementation (DEAP's C version fails silently in some envs)
   187:     front_values = np.array([ind.fitness.values for ind in nd_front])
   188:     ref = np.array(ref_point, dtype=np.float64)
   189:     # Filter out points not dominated by ref
   190:     mask = np.all(front_values < ref, axis=1)
   191:     front_values = front_values[mask]
   192:     if len(front_values) == 0:
   193:         return 0.0
   194:     n_obj = front_values.shape[1]
   195:     if n_obj == 2:
   196:         return _hv_2d(front_values, ref)
   197:     elif n_obj == 3:
   198:         return _hv_3d(front_values, ref)
   199:     return 0.0
   200: 
   201: 
   202: def compute_spread(front_values: np.ndarray) -> float:
   203:     """Compute spread (Delta) metric for a 2D front.
   204: 
   205:     Measures the extent and uniformity of the Pareto front approximation.
   206:     Lower is better. For >2 objectives, returns average pairwise distance std.
   207:     """
   208:     if len(front_values) < 2:
   209:         return float("inf")
   210: 
   211:     n_obj = front_values.shape[1]
   212:     if n_obj == 2:
   213:         # Sort by first objective
   214:         sorted_idx = np.argsort(front_values[:, 0])
   215:         sorted_front = front_values[sorted_idx]
   216:         # Consecutive distances
   217:         dists = np.sqrt(np.sum(np.diff(sorted_front, axis=0) ** 2, axis=1))
   218:         if len(dists) == 0:
   219:             return float("inf")
   220:         d_mean = np.mean(dists)
   221:         if d_mean < 1e-12:
   222:             return float("inf")
   223:         spread = np.sum(np.abs(dists - d_mean)) / (len(dists) * d_mean)
   224:         return float(spread)
   225:     else:
   226:         # For many-objective: use spacing metric
   227:         from scipy.spatial.distance import cdist
   228: 
   229:         dist_matrix = cdist(front_values, front_values)
   230:         np.fill_diagonal(dist_matrix, np.inf)
   231:         min_dists = np.min(dist_matrix, axis=1)
   232:         d_mean = np.mean(min_dists)
   233:         if d_mean < 1e-12:
   234:             return float("inf")
   235:         spread = np.sqrt(np.mean((min_dists - d_mean) ** 2)) / d_mean
   236:         return float(spread)
   237: 
   238: 
   239: def make_individual(n_var, bounds, ind_class):
   240:     """Create a random individual within bounds."""
   241:     lo, hi = bounds
   242:     return ind_class([random.uniform(lo, hi) for _ in range(n_var)])
   243: 
   244: 
   245: def evaluate(individual, func):
   246:     """Evaluate an individual on the benchmark function."""
   247:     return func(individual)
   248: 
   249: 
   250: def bounded_crossover(ind1, ind2, eta, low, up):
   251:     """Simulated Binary Crossover (SBX) with bounds."""
   252:     tools.cxSimulatedBinaryBounded(ind1, ind2, eta=eta, low=low, up=up)
   253:     return ind1, ind2
   254: 
   255: 
   256: def bounded_mutation(individual, eta, low, up, indpb):
   257:     """Polynomial mutation with bounds."""
   258:     tools.mutPolynomialBounded(individual, eta=eta, low=low, up=up, indpb=indpb)
   259:     return (individual,)
   260: 
   261: 
   262: def get_nondominated(population):
   263:     """Extract the first non-dominated front from the population."""
   264:     pareto_fronts = tools.sortNondominated(population, len(population), first_front_only=True)
   265:     return pareto_fronts[0]
   266: 
   267: 
   268: def compute_crowding_distance(individuals):
   269:     """Compute crowding distance for a set of individuals."""
   270:     if len(individuals) <= 2:
   271:         for ind in individuals:
   272:             ind.fitness.crowding_dist = float("inf")
   273:         return
   274:     n_obj = len(individuals[0].fitness.values)
   275:     for ind in individuals:
   276:         ind.fitness.crowding_dist = 0.0
   277:     for m in range(n_obj):
   278:         individuals.sort(key=lambda x: x.fitness.values[m])
   279:         individuals[0].fitness.crowding_dist = float("inf")
   280:         individuals[-1].fitness.crowding_dist = float("inf")
   281:         f_max = individuals[-1].fitness.values[m]
   282:         f_min = individuals[0].fitness.values[m]
   283:         if f_max - f_min < 1e-12:
   284:             continue
   285:         for i in range(1, len(individuals) - 1):
   286:             individuals[i].fitness.crowding_dist += (
   287:                 individuals[i + 1].fitness.values[m] - individuals[i - 1].fitness.values[m]
   288:             ) / (f_max - f_min)
   289: 
   290: 
   291: # ================================================================
   292: # EDITABLE — Custom multi-objective evolutionary strategy (lines 297 to 446)
   293: # The agent modifies ONLY this section.
   294: # ================================================================
   295: 
   296: 
   297: class CustomMOEA:
   298:     """Custom multi-objective evolutionary algorithm.
   299: 
   300:     The agent should implement a novel evolutionary strategy for multi-objective
   301:     optimization. The algorithm operates on a population of individuals, each
   302:     with a fitness consisting of multiple objective values (all minimized).
   303: 
   304:     Available DEAP utilities (already imported):
   305:         - tools.sortNondominated(pop, k) -> list of fronts
   306:         - tools.selTournamentDCD(pop, k) -> selected individuals
   307:         - tools.cxSimulatedBinaryBounded(ind1, ind2, eta, low, up)
   308:         - tools.mutPolynomialBounded(ind, eta, low, up, indpb)
   309:         - tools.uniform_reference_points(nobj, p) -> reference points array
   310:         - compute_crowding_distance(individuals) -> sets .fitness.crowding_dist
   311:         - get_nondominated(population) -> first front
   312: 
   313:     Individual interface:
   314:         ind.fitness.values -> tuple of objective values (all minimized)
   315:         ind.fitness.dominates(other.fitness) -> bool
   316:         ind.fitness.valid -> bool (True if evaluated)
   317: 
   318:     Args:
   319:         pop_size: population size
   320:         n_obj: number of objectives
   321:         n_var: number of decision variables
   322:         bounds: (low, high) for all variables
   323:         cx_eta: SBX crossover distribution index (default 20)
   324:         mut_eta: polynomial mutation distribution index (default 20)
   325:         mut_prob: per-variable mutation probability (default 1/n_var)
   326:     """
   327: 
   328:     def __init__(
   329:         self,
   330:         pop_size: int,
   331:         n_obj: int,
   332:         n_var: int,
   333:         bounds: Tuple[float, float],
   334:         cx_eta: float = 20.0,
   335:         mut_eta: float = 20.0,
   336:         mut_prob: Optional[float] = None,
   337:     ):
   338:         self.pop_size = pop_size
   339:         self.n_obj = n_obj
   340:         self.n_var = n_var
   341:         self.bounds = bounds
   342:         self.cx_eta = cx_eta
   343:         self.mut_eta = mut_eta
   344:         self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var
   345: 
   346:     def select(self, population: list, k: int) -> list:
   347:         """Select k parents from the population for mating.
   348: 
   349:         Default: binary tournament selection based on non-domination rank
   350:         and crowding distance (NSGA-II style). Replace with a better strategy.
   351: 
   352:         Args:
   353:             population: current population (list of Individuals)
   354:             k: number of parents to select
   355:         Returns:
   356:             list of k selected individuals (copies)
   357:         """
   358:         # Assign crowding distances for tournament selection
   359:         fronts = tools.sortNondominated(population, len(population), first_front_only=False)
   360:         for front in fronts:
   361:             compute_crowding_distance(front)
   362:         return tools.selTournamentDCD(population, k)
   363: 
   364:     def vary(self, parents: list) -> list:
   365:         """Apply crossover and mutation to produce offspring.
   366: 
   367:         Default: SBX crossover (probability 0.9) + polynomial mutation.
   368:         Replace or augment with novel variation operators.
   369: 
   370:         Args:
   371:             parents: list of selected parent individuals
   372:         Returns:
   373:             list of offspring individuals (fitness invalidated)
   374:         """
   375:         offspring = [deepcopy(ind) for ind in parents]
   376:         lo, hi = self.bounds
   377: 
   378:         # Pairwise crossover
   379:         for i in range(0, len(offspring) - 1, 2):
   380:             if random.random() < 0.9:
   381:                 tools.cxSimulatedBinaryBounded(
   382:                     offspring[i], offspring[i + 1],
   383:                     eta=self.cx_eta, low=lo, up=hi,
   384:                 )
   385:                 del offspring[i].fitness.values
   386:                 del offspring[i + 1].fitness.values
   387: 
   388:         # Mutation
   389:         for ind in offspring:
   390:             if random.random() < 1.0:
   391:                 tools.mutPolynomialBounded(
   392:                     ind, eta=self.mut_eta, low=lo, up=hi, indpb=self.mut_prob,
   393:                 )
   394:                 del ind.fitness.values
   395: 
   396:         return offspring
   397: 
   398:     def survive(self, population: list, offspring: list) -> list:
   399:         """Environmental selection: choose next generation from combined pool.
   400: 
   401:         Default: NSGA-II survival — non-dominated sorting + crowding distance.
   402:         Replace with a better environmental selection mechanism.
   403: 
   404:         Args:
   405:             population: current population
   406:             offspring: newly generated offspring
   407:         Returns:
   408:             list of pop_size individuals for the next generation
   409:         """
   410:         combined = population + offspring
   411: 
   412:         # Non-dominated sorting
   413:         fronts = tools.sortNondominated(combined, self.pop_size, first_front_only=False)
   414: 
   415:         next_gen = []
   416:         for front in fronts:
   417:             if len(next_gen) + len(front) <= self.pop_size:
   418:                 next_gen.extend(front)
   419:             else:
   420:                 # Fill remaining slots using crowding distance
   421:                 remaining = self.pop_size - len(next_gen)
   422:                 compute_crowding_distance(front)
   423:                 front.sort(key=lambda x: x.fitness.crowding_dist, reverse=True)
   424:                 next_gen.extend(front[:remaining])
   425:                 break
   426: 
   427:         return next_gen
   428: 
   429:     def on_generation(self, gen: int, population: list):
   430:         """Optional callback at the end of each generation.
   431: 
   432:         Can be used for adaptive parameter updates, archive maintenance, etc.
   433:         Default: no-op.
   434: 
   435:         Args:
   436:             gen: current generation number (1-indexed)
   437:             population: current population after survival selection
   438:         """
   439:         pass
   440: 
   441: 
   442: # ================================================================
   443: # FIXED — Main evolution loop and evaluation (do not modify below)
   444: # ================================================================
   445: 
   446: 
   447: def run_moea(problem_name: str, seed: int, output_dir: str):
   448:     """Run the custom MOEA on a benchmark problem."""
   449:     cfg = PROBLEMS[problem_name]
   450:     func = cfg["func"]
   451:     n_var = cfg["n_var"]
   452:     n_obj = cfg["n_obj"]
   453:     bounds = cfg["bounds"]
   454:     pop_size = cfg["pop_size"]
   455:     n_gen = cfg["n_gen"]
   456:     ref_point = cfg["ref_point"]
   457: 
   458:     # Set seeds
   459:     random.seed(seed)
   460:     np.random.seed(seed)
   461: 
   462:     # Determine individual class based on number of objectives
   463:     ind_class = creator.Individual3 if n_obj == 3 else creator.Individual
   464: 
   465:     # Initialize algorithm
   466:     moea = CustomMOEA(
   467:         pop_size=pop_size,
   468:         n_obj=n_obj,
   469:         n_var=n_var,
   470:         bounds=bounds,
   471:     )
   472: 
   473:     # Create initial population
   474:     population = [make_individual(n_var, bounds, ind_class) for _ in range(pop_size)]
   475: 
   476:     # Evaluate initial population
   477:     for ind in population:
   478:         ind.fitness.values = evaluate(ind, func)
   479: 
   480:     # Generate reference Pareto front for IGD
   481:     pf_ref = generate_pareto_front(problem_name, n_points=500)
   482: 
   483:     # Track metrics over generations
   484:     hv_history = []
   485:     igd_history = []
   486: 
   487:     for gen in range(1, n_gen + 1):
   488:         # Parent selection
   489:         parents = moea.select(population, pop_size)
   490: 
   491:         # Variation (crossover + mutation)
   492:         offspring = moea.vary(parents)
   493: 
   494:         # Evaluate offspring
   495:         for ind in offspring:
   496:             if not ind.fitness.valid:
   497:                 ind.fitness.values = evaluate(ind, func)
   498: 
   499:         # Environmental selection (survival)
   500:         population = moea.survive(population, offspring)

[truncated: showing at most 500 lines / 60000 bytes from deap/custom_moea.py]
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `nsga2` baseline — editable region  [READ-ONLY — reference implementation]

In `deap/custom_moea.py`:

```python
Lines 297–361:
   294: # ================================================================
   295: 
   296: 
   297: 
   298: class CustomMOEA:
   299:     """NSGA-II: Non-dominated Sorting Genetic Algorithm II."""
   300: 
   301:     def __init__(self, pop_size, n_obj, n_var, bounds, cx_eta=20.0, mut_eta=20.0, mut_prob=None):
   302:         self.pop_size = pop_size
   303:         self.n_obj = n_obj
   304:         self.n_var = n_var
   305:         self.bounds = bounds
   306:         self.cx_eta = cx_eta
   307:         self.mut_eta = mut_eta
   308:         self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var
   309: 
   310:     def select(self, population, k):
   311:         """Binary tournament selection with crowding distance."""
   312:         fronts = tools.sortNondominated(population, len(population), first_front_only=False)
   313:         for front in fronts:
   314:             compute_crowding_distance(front)
   315:         return tools.selTournamentDCD(population, k)
   316: 
   317:     def vary(self, parents):
   318:         """SBX crossover + polynomial mutation."""
   319:         offspring = [deepcopy(ind) for ind in parents]
   320:         lo, hi = self.bounds
   321: 
   322:         for i in range(0, len(offspring) - 1, 2):
   323:             if random.random() < 0.9:
   324:                 tools.cxSimulatedBinaryBounded(
   325:                     offspring[i], offspring[i + 1],
   326:                     eta=self.cx_eta, low=lo, up=hi,
   327:                 )
   328:                 del offspring[i].fitness.values
   329:                 del offspring[i + 1].fitness.values
   330: 
   331:         for ind in offspring:
   332:             if random.random() < 1.0:
   333:                 tools.mutPolynomialBounded(
   334:                     ind, eta=self.mut_eta, low=lo, up=hi, indpb=self.mut_prob,
   335:                 )
   336:                 del ind.fitness.values
   337: 
   338:         return offspring
   339: 
   340:     def survive(self, population, offspring):
   341:         """NSGA-II survival: non-dominated sorting + crowding distance."""
   342:         combined = population + offspring
   343:         fronts = tools.sortNondominated(combined, self.pop_size, first_front_only=False)
   344: 
   345:         next_gen = []
   346:         for front in fronts:
   347:             if len(next_gen) + len(front) <= self.pop_size:
   348:                 next_gen.extend(front)
   349:             else:
   350:                 remaining = self.pop_size - len(next_gen)
   351:                 compute_crowding_distance(front)
   352:                 front.sort(key=lambda x: x.fitness.crowding_dist, reverse=True)
   353:                 next_gen.extend(front[:remaining])
   354:                 break
   355: 
   356:         return next_gen
   357: 
   358:     def on_generation(self, gen, population):
   359:         pass
   360: 
   361: 
   362: # ================================================================
   363: # FIXED — Main evolution loop and evaluation (do not modify below)
   364: # ================================================================
```

### `moead` baseline — editable region  [READ-ONLY — reference implementation]

In `deap/custom_moea.py`:

```python
Lines 297–418:
   294: # ================================================================
   295: 
   296: 
   297: 
   298: class CustomMOEA:
   299:     """MOEA/D: Multi-Objective Evolutionary Algorithm Based on Decomposition."""
   300: 
   301:     def __init__(self, pop_size, n_obj, n_var, bounds, cx_eta=20.0, mut_eta=20.0, mut_prob=None):
   302:         self.pop_size = pop_size
   303:         self.n_obj = n_obj
   304:         self.n_var = n_var
   305:         self.bounds = bounds
   306:         self.cx_eta = cx_eta
   307:         self.mut_eta = mut_eta
   308:         self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var
   309:         self.T = 20  # neighborhood size
   310:         self.delta = 0.9  # probability of selecting from neighborhood
   311: 
   312:         # Generate weight vectors
   313:         self.weights = self._generate_weights(pop_size, n_obj)
   314:         self.pop_size = len(self.weights)  # adjust to actual number of weight vectors
   315: 
   316:         # Compute neighborhoods
   317:         self.neighbors = self._compute_neighborhoods()
   318: 
   319:         # Ideal point (updated during search)
   320:         self.z_star = None
   321: 
   322:     def _generate_weights(self, n, n_obj):
   323:         """Generate uniformly distributed weight vectors."""
   324:         if n_obj == 2:
   325:             weights = []
   326:             for i in range(n):
   327:                 w1 = i / max(n - 1, 1)
   328:                 weights.append([w1, 1.0 - w1])
   329:             return np.array(weights)
   330:         else:
   331:             # Use DEAP's uniform reference points for 3+ objectives
   332:             ref_points = tools.uniform_reference_points(n_obj, p=12)
   333:             return np.array(ref_points)
   334: 
   335:     def _compute_neighborhoods(self):
   336:         """Compute T-nearest weight vector neighborhoods."""
   337:         from scipy.spatial.distance import cdist
   338:         dist_matrix = cdist(self.weights, self.weights)
   339:         neighbors = []
   340:         for i in range(len(self.weights)):
   341:             idx = np.argsort(dist_matrix[i])[:self.T]
   342:             neighbors.append(idx.tolist())
   343:         return neighbors
   344: 
   345:     def _tchebycheff(self, fitness_values, weight, z_star):
   346:         """Tchebycheff scalarization."""
   347:         return max(weight[j] * abs(fitness_values[j] - z_star[j])
   348:                    for j in range(self.n_obj))
   349: 
   350:     def select(self, population, k):
   351:         """MOEA/D doesn't use standard selection — return population as-is."""
   352:         return [deepcopy(ind) for ind in population]
   353: 
   354:     def vary(self, parents):
   355:         """Generate one offspring per subproblem using neighborhood mating."""
   356:         offspring = []
   357:         lo, hi = self.bounds
   358: 
   359:         for i in range(len(parents)):
   360:             # Select mating pool (neighborhood or whole population)
   361:             if random.random() < self.delta:
   362:                 pool = [parents[j] for j in self.neighbors[i % len(self.neighbors)]]
   363:             else:
   364:                 pool = parents
   365: 
   366:             # Select two parents from pool
   367:             p1, p2 = random.sample(range(len(pool)), 2)
   368:             child = deepcopy(pool[p1])
   369: 
   370:             # SBX crossover
   371:             mate = deepcopy(pool[p2])
   372:             if random.random() < 1.0:
   373:                 tools.cxSimulatedBinaryBounded(child, mate, eta=self.cx_eta, low=lo, up=hi)
   374: 
   375:             # Polynomial mutation
   376:             tools.mutPolynomialBounded(child, eta=self.mut_eta, low=lo, up=hi, indpb=self.mut_prob)
   377:             del child.fitness.values
   378:             offspring.append(child)
   379: 
   380:         return offspring
   381: 
   382:     def survive(self, population, offspring):
   383:         """MOEA/D survival: update subproblems using Tchebycheff decomposition."""
   384:         # Update ideal point
   385:         all_inds = [ind for ind in population + offspring if ind.fitness.valid]
   386:         if not all_inds:
   387:             return population
   388: 
   389:         if self.z_star is None:
   390:             self.z_star = [float('inf')] * self.n_obj
   391:         for ind in all_inds:
   392:             for j in range(self.n_obj):
   393:                 if ind.fitness.values[j] < self.z_star[j]:
   394:                     self.z_star[j] = ind.fitness.values[j]
   395: 
   396:         # Update each subproblem
   397:         next_gen = list(population)
   398:         for i in range(min(len(offspring), len(self.weights))):
   399:             child = offspring[i]
   400:             if not child.fitness.valid:
   401:                 continue
   402: 
   403:             # Update neighbors
   404:             neighbors_idx = self.neighbors[i % len(self.neighbors)]
   405:             for j_idx in neighbors_idx:
   406:                 if j_idx >= len(next_gen):
   407:                     continue
   408:                 g_child = self._tchebycheff(child.fitness.values, self.weights[j_idx], self.z_star)
   409:                 g_current = self._tchebycheff(next_gen[j_idx].fitness.values, self.weights[j_idx], self.z_star)
   410:                 if g_child < g_current:
   411:                     next_gen[j_idx] = deepcopy(child)
   412: 
   413:         return next_gen[:self.pop_size]
   414: 
   415:     def on_generation(self, gen, population):
   416:         pass
   417: 
   418: 
   419: # ================================================================
   420: # FIXED — Main evolution loop and evaluation (do not modify below)
   421: # ================================================================
```

### `spea2` baseline — editable region  [READ-ONLY — reference implementation]

In `deap/custom_moea.py`:

```python
Lines 297–367:
   294: # ================================================================
   295: 
   296: 
   297: 
   298: class CustomMOEA:
   299:     """SPEA2: Strength Pareto Evolutionary Algorithm 2."""
   300: 
   301:     def __init__(self, pop_size, n_obj, n_var, bounds, cx_eta=20.0, mut_eta=20.0, mut_prob=None):
   302:         self.pop_size = pop_size
   303:         self.n_obj = n_obj
   304:         self.n_var = n_var
   305:         self.bounds = bounds
   306:         self.cx_eta = cx_eta
   307:         self.mut_eta = mut_eta
   308:         self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var
   309:         self.archive = []
   310: 
   311:     def select(self, population, k):
   312:         """Binary tournament selection using SPEA2 fitness from archive."""
   313:         # Use archive for selection if available, otherwise population
   314:         pool = self.archive if self.archive else population
   315:         # Binary tournament on dominance
   316:         selected = []
   317:         for _ in range(k):
   318:             i1, i2 = random.sample(range(len(pool)), 2)
   319:             a, b = pool[i1], pool[i2]
   320:             if a.fitness.dominates(b.fitness):
   321:                 selected.append(deepcopy(a))
   322:             elif b.fitness.dominates(a.fitness):
   323:                 selected.append(deepcopy(b))
   324:             else:
   325:                 selected.append(deepcopy(random.choice([a, b])))
   326:         return selected
   327: 
   328:     def vary(self, parents):
   329:         """SBX crossover + polynomial mutation."""
   330:         offspring = [deepcopy(ind) for ind in parents]
   331:         lo, hi = self.bounds
   332: 
   333:         for i in range(0, len(offspring) - 1, 2):
   334:             if random.random() < 0.9:
   335:                 tools.cxSimulatedBinaryBounded(
   336:                     offspring[i], offspring[i + 1],
   337:                     eta=self.cx_eta, low=lo, up=hi,
   338:                 )
   339:                 del offspring[i].fitness.values
   340:                 del offspring[i + 1].fitness.values
   341: 
   342:         for ind in offspring:
   343:             if random.random() < 1.0:
   344:                 tools.mutPolynomialBounded(
   345:                     ind, eta=self.mut_eta, low=lo, up=hi, indpb=self.mut_prob,
   346:                 )
   347:                 del ind.fitness.values
   348: 
   349:         return offspring
   350: 
   351:     def survive(self, population, offspring):
   352:         """SPEA2 survival: strength fitness + kNN density truncation."""
   353:         combined = population + offspring
   354: 
   355:         # Use DEAP's built-in SPEA2 selection
   356:         selected = tools.selSPEA2(combined, self.pop_size)
   357: 
   358:         # Update archive with non-dominated solutions
   359:         nd = get_nondominated(selected)
   360:         self.archive = [deepcopy(ind) for ind in nd[:self.pop_size]]
   361: 
   362:         return selected
   363: 
   364:     def on_generation(self, gen, population):
   365:         pass
   366: 
   367: 
   368: # ================================================================
   369: # FIXED — Main evolution loop and evaluation (do not modify below)
   370: # ================================================================
```

### `nsga3` baseline — editable region  [READ-ONLY — reference implementation]

In `deap/custom_moea.py`:

```python
Lines 297–357:
   294: # ================================================================
   295: 
   296: 
   297: 
   298: class CustomMOEA:
   299:     """NSGA-III: Non-dominated Sorting Genetic Algorithm III."""
   300: 
   301:     def __init__(self, pop_size, n_obj, n_var, bounds, cx_eta=20.0, mut_eta=20.0, mut_prob=None):
   302:         self.pop_size = pop_size
   303:         self.n_obj = n_obj
   304:         self.n_var = n_var
   305:         self.bounds = bounds
   306:         self.cx_eta = cx_eta
   307:         self.mut_eta = mut_eta
   308:         self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var
   309: 
   310:         # Generate reference points
   311:         if n_obj == 2:
   312:             p = pop_size - 1  # number of divisions
   313:             self.ref_points = tools.uniform_reference_points(n_obj, p=p)
   314:         else:
   315:             self.ref_points = tools.uniform_reference_points(n_obj, p=12)
   316: 
   317:     def select(self, population, k):
   318:         """Random shuffle selection (NSGA-III relies on survive for diversity)."""
   319:         selected = [deepcopy(ind) for ind in population]
   320:         random.shuffle(selected)
   321:         return selected[:k]
   322: 
   323:     def vary(self, parents):
   324:         """SBX crossover + polynomial mutation."""
   325:         offspring = [deepcopy(ind) for ind in parents]
   326:         lo, hi = self.bounds
   327: 
   328:         for i in range(0, len(offspring) - 1, 2):
   329:             if random.random() < 1.0:
   330:                 tools.cxSimulatedBinaryBounded(
   331:                     offspring[i], offspring[i + 1],
   332:                     eta=self.cx_eta, low=lo, up=hi,
   333:                 )
   334:                 del offspring[i].fitness.values
   335:                 del offspring[i + 1].fitness.values
   336: 
   337:         for ind in offspring:
   338:             if random.random() < 1.0:
   339:                 tools.mutPolynomialBounded(
   340:                     ind, eta=self.mut_eta, low=lo, up=hi, indpb=self.mut_prob,
   341:                 )
   342:                 del ind.fitness.values
   343: 
   344:         return offspring
   345: 
   346:     def survive(self, population, offspring):
   347:         """NSGA-III survival: reference-point-based selection."""
   348:         combined = population + offspring
   349: 
   350:         # Use DEAP's built-in NSGA-III selection
   351:         selected = tools.selNSGA3(combined, self.pop_size, self.ref_points)
   352:         return selected
   353: 
   354:     def on_generation(self, gen, population):
   355:         pass
   356: 
   357: 
   358: # ================================================================
   359: # FIXED — Main evolution loop and evaluation (do not modify below)
   360: # ================================================================
```

### `rvea` baseline — editable region  [READ-ONLY — reference implementation]

In `deap/custom_moea.py`:

```python
Lines 297–449:
   294: # ================================================================
   295: 
   296: 
   297: 
   298: class CustomMOEA:
   299:     """RVEA: Reference Vector Guided Evolutionary Algorithm."""
   300: 
   301:     def __init__(self, pop_size, n_obj, n_var, bounds, cx_eta=20.0, mut_eta=20.0, mut_prob=None):
   302:         self.pop_size = pop_size
   303:         self.n_obj = n_obj
   304:         self.n_var = n_var
   305:         self.bounds = bounds
   306:         self.cx_eta = cx_eta
   307:         self.mut_eta = mut_eta
   308:         self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var
   309:         self.alpha = 2.0  # penalty parameter for APD
   310:         self.fr = 0.1  # frequency of reference vector adaptation
   311: 
   312:         # Generate initial reference vectors
   313:         if n_obj == 2:
   314:             p = pop_size - 1
   315:             self.ref_vectors = np.array(tools.uniform_reference_points(n_obj, p=p))
   316:         else:
   317:             self.ref_vectors = np.array(tools.uniform_reference_points(n_obj, p=12))
   318:         self.ref_vectors_initial = self.ref_vectors.copy()
   319: 
   320:         # Normalize reference vectors to unit length
   321:         norms = np.linalg.norm(self.ref_vectors, axis=1, keepdims=True)
   322:         norms[norms < 1e-12] = 1e-12
   323:         self.ref_vectors = self.ref_vectors / norms
   324: 
   325:     def _angle_penalized_distance(self, fitness_values, gen, max_gen):
   326:         """Compute angle-penalized distance for each individual to its closest reference vector."""
   327:         F = np.array(fitness_values)
   328:         n = len(F)
   329:         n_ref = len(self.ref_vectors)
   330: 
   331:         if n == 0:
   332:             return np.array([]), np.array([])
   333: 
   334:         # Translate objectives (subtract ideal point)
   335:         z_min = np.min(F, axis=0)
   336:         F_translated = F - z_min + 1e-12
   337: 
   338:         # Compute angles between each individual and each reference vector
   339:         # cos(theta) = (F . v) / (||F|| * ||v||)
   340:         F_norms = np.linalg.norm(F_translated, axis=1, keepdims=True)
   341:         F_norms[F_norms < 1e-12] = 1e-12
   342:         F_normalized = F_translated / F_norms
   343: 
   344:         # Cosine similarity
   345:         cos_angles = F_normalized @ self.ref_vectors.T  # (n, n_ref)
   346:         cos_angles = np.clip(cos_angles, -1.0, 1.0)
   347:         angles = np.arccos(cos_angles)  # (n, n_ref)
   348: 
   349:         # Associate each individual with closest reference vector
   350:         associations = np.argmin(angles, axis=1)  # (n,)
   351:         min_angles = angles[np.arange(n), associations]  # (n,)
   352: 
   353:         # Compute convergence (distance along reference vector)
   354:         convergence = F_norms.flatten()
   355: 
   356:         # Angle penalty that increases over generations
   357:         gamma = self.alpha * (gen / max(max_gen, 1)) ** 2
   358: 
   359:         # APD = convergence * (1 + gamma * angle)
   360:         apd = convergence * (1.0 + gamma * min_angles)
   361: 
   362:         return apd, associations
   363: 
   364:     def select(self, population, k):
   365:         """Random mating selection."""
   366:         selected = [deepcopy(ind) for ind in population]
   367:         random.shuffle(selected)
   368:         return selected[:k]
   369: 
   370:     def vary(self, parents):
   371:         """SBX crossover + polynomial mutation."""
   372:         offspring = [deepcopy(ind) for ind in parents]
   373:         lo, hi = self.bounds
   374: 
   375:         for i in range(0, len(offspring) - 1, 2):
   376:             if random.random() < 1.0:
   377:                 tools.cxSimulatedBinaryBounded(
   378:                     offspring[i], offspring[i + 1],
   379:                     eta=self.cx_eta, low=lo, up=hi,
   380:                 )
   381:                 del offspring[i].fitness.values
   382:                 del offspring[i + 1].fitness.values
   383: 
   384:         for ind in offspring:
   385:             if random.random() < 1.0:
   386:                 tools.mutPolynomialBounded(
   387:                     ind, eta=self.mut_eta, low=lo, up=hi, indpb=self.mut_prob,
   388:                 )
   389:                 del ind.fitness.values
   390: 
   391:         return offspring
   392: 
   393:     def survive(self, population, offspring):
   394:         """RVEA survival: angle-penalized distance based selection."""
   395:         combined = population + offspring
   396:         valid = [ind for ind in combined if ind.fitness.valid]
   397: 
   398:         if len(valid) <= self.pop_size:
   399:             return valid
   400: 
   401:         fitness_values = [ind.fitness.values for ind in valid]
   402:         # Use a large gen estimate based on problem config
   403:         max_gen = 400
   404:         gen_estimate = getattr(self, '_current_gen', max_gen // 2)
   405:         apd, associations = self._angle_penalized_distance(fitness_values, gen_estimate, max_gen)
   406: 
   407:         # Select the best individual per reference vector (lowest APD)
   408:         selected_indices = set()
   409:         n_ref = len(self.ref_vectors)
   410:         for v in range(n_ref):
   411:             mask = np.where(associations == v)[0]
   412:             if len(mask) > 0:
   413:                 best_idx = mask[np.argmin(apd[mask])]
   414:                 selected_indices.add(best_idx)
   415: 
   416:         # If not enough, fill with best remaining by APD
   417:         if len(selected_indices) < self.pop_size:
   418:             remaining = [i for i in range(len(valid)) if i not in selected_indices]
   419:             remaining.sort(key=lambda i: apd[i])
   420:             for i in remaining:
   421:                 selected_indices.add(i)
   422:                 if len(selected_indices) >= self.pop_size:
   423:                     break
   424: 
   425:         # If too many (more ref vectors than pop_size), truncate by APD
   426:         selected_list = sorted(selected_indices, key=lambda i: apd[i])[:self.pop_size]
   427:         return [valid[i] for i in selected_list]
   428: 
   429:     def on_generation(self, gen, population):
   430:         """Adapt reference vectors periodically."""
   431:         self._current_gen = gen
   432: 
   433:         # Reference vector adaptation
   434:         max_gen = 400
   435:         if gen % max(1, int(self.fr * max_gen)) == 0 and len(population) > 0:
   436:             fitness_values = np.array([ind.fitness.values for ind in population if ind.fitness.valid])
   437:             if len(fitness_values) > 0:
   438:                 z_max = np.max(fitness_values, axis=0)
   439:                 z_min = np.min(fitness_values, axis=0)
   440:                 scale = z_max - z_min
   441:                 scale[scale < 1e-12] = 1.0
   442: 
   443:                 # Scale reference vectors
   444:                 self.ref_vectors = self.ref_vectors_initial * scale
   445:                 norms = np.linalg.norm(self.ref_vectors, axis=1, keepdims=True)
   446:                 norms[norms < 1e-12] = 1e-12
   447:                 self.ref_vectors = self.ref_vectors / norms
   448: 
   449: 
   450: # ================================================================
   451: # FIXED — Main evolution loop and evaluation (do not modify below)
   452: # ================================================================
```

### `agemoea` baseline — editable region  [READ-ONLY — reference implementation]

In `deap/custom_moea.py`:

```python
Lines 297–473:
   294: # ================================================================
   295: 
   296: 
   297: 
   298: class CustomMOEA:
   299:     """AGE-MOEA: Adaptive Geometry Estimation based MOEA."""
   300: 
   301:     def __init__(self, pop_size, n_obj, n_var, bounds, cx_eta=20.0, mut_eta=20.0, mut_prob=None):
   302:         self.pop_size = pop_size
   303:         self.n_obj = n_obj
   304:         self.n_var = n_var
   305:         self.bounds = bounds
   306:         self.cx_eta = cx_eta
   307:         self.mut_eta = mut_eta
   308:         self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var
   309: 
   310:     def _estimate_geometry(self, front_values):
   311:         """Estimate the geometry parameter p of the Pareto front.
   312: 
   313:         Uses the relationship between Lp-norm and front shape:
   314:         p=1: linear front (like DTLZ1)
   315:         p=2: spherical front (like DTLZ2)
   316:         p->inf: rectangular front
   317:         """
   318:         if len(front_values) < 2 or self.n_obj < 2:
   319:             return 1.0
   320: 
   321:         F = np.array(front_values)
   322: 
   323:         # Normalize objectives
   324:         z_min = np.min(F, axis=0)
   325:         z_max = np.max(F, axis=0)
   326:         scale = z_max - z_min
   327:         scale[scale < 1e-12] = 1.0
   328:         F_norm = (F - z_min) / scale
   329: 
   330:         # Find extreme points (closest to axes)
   331:         extremes = []
   332:         for m in range(self.n_obj):
   333:             # Point with smallest value on objective m
   334:             idx = np.argmin(F_norm[:, m])
   335:             extremes.append(F_norm[idx])
   336: 
   337:         if len(extremes) < 2:
   338:             return 1.0
   339: 
   340:         # Estimate p from extreme points
   341:         # For an Lp-norm sphere of radius r: sum(|x_i/r|^p) = 1
   342:         # Use the median point on the front to estimate p
   343:         median_idx = len(F_norm) // 2
   344:         median_point = np.sort(F_norm, axis=0)[median_idx]
   345: 
   346:         # Avoid zero/negative values
   347:         median_point = np.maximum(median_point, 1e-8)
   348: 
   349:         # Binary search for p
   350:         p_low, p_high = 0.1, 20.0
   351:         for _ in range(50):
   352:             p_mid = (p_low + p_high) / 2
   353:             lp_val = np.sum(median_point ** p_mid)
   354:             if lp_val > 1.0:
   355:                 p_low = p_mid
   356:             else:
   357:                 p_high = p_mid
   358:         p = (p_low + p_high) / 2
   359:         return max(0.1, min(p, 20.0))
   360: 
   361:     def _survival_score(self, front_values, p):
   362:         """Compute survival score based on Lp-distance-based crowding."""
   363:         F = np.array(front_values)
   364:         n = len(F)
   365:         if n <= 2:
   366:             return np.full(n, float('inf'))
   367: 
   368:         # Normalize
   369:         z_min = np.min(F, axis=0)
   370:         z_max = np.max(F, axis=0)
   371:         scale = z_max - z_min
   372:         scale[scale < 1e-12] = 1.0
   373:         F_norm = (F - z_min) / scale
   374: 
   375:         # Compute pairwise Lp-distances
   376:         scores = np.zeros(n)
   377:         for i in range(n):
   378:             dists = []
   379:             for j in range(n):
   380:                 if i == j:
   381:                     continue
   382:                 diff = np.abs(F_norm[i] - F_norm[j])
   383:                 lp_dist = np.sum(diff ** p) ** (1.0 / p)
   384:                 dists.append(lp_dist)
   385:             dists.sort()
   386:             # Use nearest neighbor distance as diversity score
   387:             if dists:
   388:                 scores[i] = dists[0]
   389:             else:
   390:                 scores[i] = 0.0
   391: 
   392:         return scores
   393: 
   394:     def select(self, population, k):
   395:         """Binary tournament selection based on non-domination rank."""
   396:         fronts = tools.sortNondominated(population, len(population), first_front_only=False)
   397:         # Assign rank
   398:         for rank, front in enumerate(fronts):
   399:             for ind in front:
   400:                 ind.fitness.crowding_dist = 0.0  # reset
   401:                 ind._rank = rank
   402:         # Tournament
   403:         selected = []
   404:         for _ in range(k):
   405:             i1, i2 = random.sample(range(len(population)), 2)
   406:             a, b = population[i1], population[i2]
   407:             if a._rank < b._rank:
   408:                 selected.append(deepcopy(a))
   409:             elif b._rank < a._rank:
   410:                 selected.append(deepcopy(b))
   411:             else:
   412:                 selected.append(deepcopy(random.choice([a, b])))
   413:         return selected
   414: 
   415:     def vary(self, parents):
   416:         """SBX crossover + polynomial mutation."""
   417:         offspring = [deepcopy(ind) for ind in parents]
   418:         lo, hi = self.bounds
   419: 
   420:         for i in range(0, len(offspring) - 1, 2):
   421:             if random.random() < 0.9:
   422:                 tools.cxSimulatedBinaryBounded(
   423:                     offspring[i], offspring[i + 1],
   424:                     eta=self.cx_eta, low=lo, up=hi,
   425:                 )
   426:                 del offspring[i].fitness.values
   427:                 del offspring[i + 1].fitness.values
   428: 
   429:         for ind in offspring:
   430:             if random.random() < 1.0:
   431:                 tools.mutPolynomialBounded(
   432:                     ind, eta=self.mut_eta, low=lo, up=hi, indpb=self.mut_prob,
   433:                 )
   434:                 del ind.fitness.values
   435: 
   436:         return offspring
   437: 
   438:     def survive(self, population, offspring):
   439:         """AGE-MOEA survival: adaptive geometry-based selection."""
   440:         combined = population + offspring
   441: 
   442:         # Non-dominated sorting
   443:         fronts = tools.sortNondominated(combined, len(combined), first_front_only=False)
   444: 
   445:         next_gen = []
   446:         for front_idx, front in enumerate(fronts):
   447:             if len(next_gen) + len(front) <= self.pop_size:
   448:                 next_gen.extend(front)
   449:             else:
   450:                 remaining = self.pop_size - len(next_gen)
   451:                 if remaining <= 0:
   452:                     break
   453: 
   454:                 # Estimate geometry from the first front
   455:                 first_front_values = [ind.fitness.values for ind in fronts[0]]
   456:                 p = self._estimate_geometry(first_front_values)
   457: 
   458:                 # Compute survival scores for the critical front
   459:                 front_values = [ind.fitness.values for ind in front]
   460:                 scores = self._survival_score(front_values, p)
   461: 
   462:                 # Select individuals with highest diversity scores
   463:                 sorted_indices = np.argsort(-scores)  # descending
   464:                 for idx in sorted_indices[:remaining]:
   465:                     next_gen.append(front[idx])
   466:                 break
   467: 
   468:         return next_gen
   469: 
   470:     def on_generation(self, gen, population):
   471:         pass
   472: 
   473: 
   474: # ================================================================
   475: # FIXED — Main evolution loop and evaluation (do not modify below)
   476: # ================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
