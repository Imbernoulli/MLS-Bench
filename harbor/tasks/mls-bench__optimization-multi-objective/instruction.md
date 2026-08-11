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
(inclusive, 1-indexed)**. Edits that change code outside these ranges — or creating new files, or
deleting whole files — will cause your submission to be invalid.

The line numbers mark an editable **region**, not a fixed line-count budget: you
may add or remove lines inside it. Only code outside the editable ranges must
stay unchanged.

- `deap/custom_moea.py`
- editable lines **159–303**




## Readable Context


### `deap/custom_moea.py`  [EDITABLE — lines 159–303 only]

```python
     1: """
     2: Multi-Objective Optimization — Custom Evolutionary Strategy Template
     3: 
     4: This script runs a complete multi-objective evolutionary algorithm on a held-out
     5: benchmark problem. The agent should implement the custom selection and variation
     6: strategy in the CustomMOEA class.
     7: 
     8: NOTE: The benchmark problem identity, its analytic true Pareto front, and the
     9: evaluation metrics are NOT part of this program. The harness pre-generates the
    10: problem to optimize (the objective functions, with their numeric configuration)
    11: and scores the final population in a separate host-side process. Your strategy
    12: only ever sees individuals with already-evaluated objective values — it never
    13: receives the problem name nor the true front.
    14: 
    15: Usage (the harness sets ENV/SEED for you):
    16:     ENV=<opaque-problem-key> SEED=42 python deap/custom_moea.py
    17: """
    18: 
    19: import argparse
    20: import base64
    21: import io
    22: import json
    23: import marshal
    24: import math
    25: import os
    26: import random
    27: import time
    28: import types
    29: import warnings
    30: from copy import deepcopy
    31: from functools import reduce
    32: from math import cos, pi, sin, sqrt
    33: from operator import mul
    34: from typing import List, Optional, Tuple
    35: 
    36: import numpy as np
    37: 
    38: from deap import base, benchmarks, creator, tools
    39: from deap.benchmarks import tools as btools
    40: 
    41: warnings.filterwarnings("ignore")
    42: 
    43: # ================================================================
    44: # FIXED — Individual types and generic utilities (do not modify)
    45: # ================================================================
    46: 
    47: # Create DEAP fitness and individual types
    48: creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    49: creator.create("Individual", list, fitness=creator.FitnessMin)
    50: 
    51: # For 3-objective problems
    52: creator.create("FitnessMin3", base.Fitness, weights=(-1.0, -1.0, -1.0))
    53: creator.create("Individual3", list, fitness=creator.FitnessMin3)
    54: 
    55: 
    56: def make_individual(n_var, bounds, ind_class):
    57:     """Create a random individual within bounds."""
    58:     lo, hi = bounds
    59:     return ind_class([random.uniform(lo, hi) for _ in range(n_var)])
    60: 
    61: 
    62: def evaluate(individual, func):
    63:     """Evaluate an individual on the (held-out) objective function."""
    64:     return func(individual)
    65: 
    66: 
    67: def bounded_crossover(ind1, ind2, eta, low, up):
    68:     """Simulated Binary Crossover (SBX) with bounds."""
    69:     tools.cxSimulatedBinaryBounded(ind1, ind2, eta=eta, low=low, up=up)
    70:     return ind1, ind2
    71: 
    72: 
    73: def bounded_mutation(individual, eta, low, up, indpb):
    74:     """Polynomial mutation with bounds."""
    75:     tools.mutPolynomialBounded(individual, eta=eta, low=low, up=up, indpb=indpb)
    76:     return (individual,)
    77: 
    78: 
    79: def get_nondominated(population):
    80:     """Extract the first non-dominated front from the population."""
    81:     pareto_fronts = tools.sortNondominated(population, len(population), first_front_only=True)
    82:     return pareto_fronts[0]
    83: 
    84: 
    85: def compute_crowding_distance(individuals):
    86:     """Compute crowding distance for a set of individuals."""
    87:     if len(individuals) <= 2:
    88:         for ind in individuals:
    89:             ind.fitness.crowding_dist = float("inf")
    90:         return
    91:     n_obj = len(individuals[0].fitness.values)
    92:     for ind in individuals:
    93:         ind.fitness.crowding_dist = 0.0
    94:     for m in range(n_obj):
    95:         individuals.sort(key=lambda x: x.fitness.values[m])
    96:         individuals[0].fitness.crowding_dist = float("inf")
    97:         individuals[-1].fitness.crowding_dist = float("inf")
    98:         f_max = individuals[-1].fitness.values[m]
    99:         f_min = individuals[0].fitness.values[m]
   100:         if f_max - f_min < 1e-12:
   101:             continue
   102:         for i in range(1, len(individuals) - 1):
   103:             individuals[i].fitness.crowding_dist += (
   104:                 individuals[i + 1].fitness.values[m] - individuals[i - 1].fitness.values[m]
   105:             ) / (f_max - f_min)
   106: 
   107: 
   108: # ================================================================
   109: # FIXED — Held-out problem spec loading (do not modify)
   110: # ================================================================
   111: #
   112: # The harness pre-generates, for the opaque problem key in ENV, a spec file
   113: # carrying the numeric problem configuration and an opaque black-box objective
   114: # evaluator f(individual) -> objectives. This program loads that spec, builds the
   115: # evaluator as a pure black box, and uses it to evaluate candidate solutions.
   116: # The problem name, the analytic Pareto front, and the metrics are NOT present
   117: # here — they live in a host-only module the agent's process cannot import. The
   118: # host-side scorer regenerates the front and computes HV/IGD/Spread.
   119: 
   120: 
   121: def _spec_dir():
   122:     d = os.environ.get("MOEA_SPEC_DIR")
   123:     if d:
   124:         return d
   125:     return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_moea_specs")
   126: 
   127: 
   128: def _load_spec(env_key, seed):
   129:     path = os.path.join(_spec_dir(), f"{env_key}_seed{seed}.json.b64")
   130:     with open(path, "r") as f:
   131:         raw = base64.b64decode(f.read())
   132:     return json.loads(raw.decode("utf-8"))
   133: 
   134: 
   135: def _build_objective(spec):
   136:     """Reconstruct the black-box objective f(individual) -> tuple from the spec.
   137: 
   138:     The evaluator is a marshalled, name-free, problem-specific code object that
   139:     inlines the objective arithmetic; it is used purely as a black box and
   140:     carries no problem identity (no name, no ``kind``) the strategy could exploit.
   141:     """
   142:     code = marshal.loads(base64.b64decode(spec["evaluator"]))
   143:     kernel = types.FunctionType(code, {"__builtins__": __builtins__}, "objective")
   144:     # No problem id in the spec; the kernel is specific to this run's problem.
   145:     n_obj = int(spec["n_obj"])
   146: 
   147:     def f(individual):
   148:         return tuple(kernel(individual, n_obj))
   149: 
   150:     return f
   151: 
   152: 
   153: # ================================================================
   154: # EDITABLE — Custom multi-objective evolutionary strategy (lines 297 to 441)
   155: # The agent modifies ONLY this section.
   156: # ================================================================
   157: 
   158: 
   159: class CustomMOEA:
   160:     """Custom multi-objective evolutionary algorithm.
   161: 
   162:     The agent should implement a novel evolutionary strategy for multi-objective
   163:     optimization. The algorithm operates on a population of individuals, each
   164:     with a fitness consisting of multiple objective values (all minimized).
   165: 
   166:     Available DEAP utilities (already imported):
   167:         - tools.sortNondominated(pop, k) -> list of fronts
   168:         - tools.selTournamentDCD(pop, k) -> selected individuals
   169:         - tools.cxSimulatedBinaryBounded(ind1, ind2, eta, low, up)
   170:         - tools.mutPolynomialBounded(ind, eta, low, up, indpb)
   171:         - tools.uniform_reference_points(nobj, p) -> reference points array
   172:         - compute_crowding_distance(individuals) -> sets .fitness.crowding_dist
   173:         - get_nondominated(population) -> first front
   174: 
   175:     Individual interface:
   176:         ind.fitness.values -> tuple of objective values (all minimized)
   177:         ind.fitness.dominates(other.fitness) -> bool
   178:         ind.fitness.valid -> bool (True if evaluated)
   179: 
   180:     Args:
   181:         pop_size: population size
   182:         n_obj: number of objectives
   183:         n_var: number of decision variables
   184:         bounds: (low, high) for all variables
   185:         cx_eta: SBX crossover distribution index (default 20)
   186:         mut_eta: polynomial mutation distribution index (default 20)
   187:         mut_prob: per-variable mutation probability (default 1/n_var)
   188:     """
   189: 
   190:     def __init__(
   191:         self,
   192:         pop_size: int,
   193:         n_obj: int,
   194:         n_var: int,
   195:         bounds: Tuple[float, float],
   196:         cx_eta: float = 20.0,
   197:         mut_eta: float = 20.0,
   198:         mut_prob: Optional[float] = None,
   199:     ):
   200:         self.pop_size = pop_size
   201:         self.n_obj = n_obj
   202:         self.n_var = n_var
   203:         self.bounds = bounds
   204:         self.cx_eta = cx_eta
   205:         self.mut_eta = mut_eta
   206:         self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var
   207: 
   208:     def select(self, population: list, k: int) -> list:
   209:         """Select k parents from the population for mating.
   210: 
   211:         Default: binary tournament selection based on non-domination rank
   212:         and crowding distance (NSGA-II style). Replace with a better strategy.
   213: 
   214:         Args:
   215:             population: current population (list of Individuals)
   216:             k: number of parents to select
   217:         Returns:
   218:             list of k selected individuals (copies)
   219:         """
   220:         # Assign crowding distances for tournament selection
   221:         fronts = tools.sortNondominated(population, len(population), first_front_only=False)
   222:         for front in fronts:
   223:             compute_crowding_distance(front)
   224:         return tools.selTournamentDCD(population, k)
   225: 
   226:     def vary(self, parents: list) -> list:
   227:         """Apply crossover and mutation to produce offspring.
   228: 
   229:         Default: SBX crossover (probability 0.9) + polynomial mutation.
   230:         Replace or augment with novel variation operators.
   231: 
   232:         Args:
   233:             parents: list of selected parent individuals
   234:         Returns:
   235:             list of offspring individuals (fitness invalidated)
   236:         """
   237:         offspring = [deepcopy(ind) for ind in parents]
   238:         lo, hi = self.bounds
   239: 
   240:         # Pairwise crossover
   241:         for i in range(0, len(offspring) - 1, 2):
   242:             if random.random() < 0.9:
   243:                 tools.cxSimulatedBinaryBounded(
   244:                     offspring[i], offspring[i + 1],
   245:                     eta=self.cx_eta, low=lo, up=hi,
   246:                 )
   247:                 del offspring[i].fitness.values
   248:                 del offspring[i + 1].fitness.values
   249: 
   250:         # Mutation
   251:         for ind in offspring:
   252:             if random.random() < 1.0:
   253:                 tools.mutPolynomialBounded(
   254:                     ind, eta=self.mut_eta, low=lo, up=hi, indpb=self.mut_prob,
   255:                 )
   256:                 del ind.fitness.values
   257: 
   258:         return offspring
   259: 
   260:     def survive(self, population: list, offspring: list) -> list:
   261:         """Environmental selection: choose next generation from combined pool.
   262: 
   263:         Default: NSGA-II survival — non-dominated sorting + crowding distance.
   264:         Replace with a better environmental selection mechanism.
   265: 
   266:         Args:
   267:             population: current population
   268:             offspring: newly generated offspring
   269:         Returns:
   270:             list of pop_size individuals for the next generation
   271:         """
   272:         combined = population + offspring
   273: 
   274:         # Non-dominated sorting
   275:         fronts = tools.sortNondominated(combined, self.pop_size, first_front_only=False)
   276: 
   277:         next_gen = []
   278:         for front in fronts:
   279:             if len(next_gen) + len(front) <= self.pop_size:
   280:                 next_gen.extend(front)
   281:             else:
   282:                 # Fill remaining slots using crowding distance
   283:                 remaining = self.pop_size - len(next_gen)
   284:                 compute_crowding_distance(front)
   285:                 front.sort(key=lambda x: x.fitness.crowding_dist, reverse=True)
   286:                 next_gen.extend(front[:remaining])
   287:                 break
   288: 
   289:         return next_gen
   290: 
   291:     def on_generation(self, gen: int, population: list):
   292:         """Optional callback at the end of each generation.
   293: 
   294:         Can be used for adaptive parameter updates, archive maintenance, etc.
   295:         Default: no-op.
   296: 
   297:         Args:
   298:             gen: current generation number (1-indexed)
   299:             population: current population after survival selection
   300:         """
   301:         pass
   302: 
   303: 
   304: # ================================================================
   305: # FIXED — Main evolution loop and prediction emit (do not modify below)
   306: # ================================================================
   307: 
   308: 
   309: def run_moea(env_key: str, seed: int, output_dir: str):
   310:     """Run the custom MOEA on the held-out benchmark problem.
   311: 
   312:     Loads the pre-generated problem spec for ``env_key``, runs the strategy, and
   313:     emits the final non-dominated population's objective values for the host-side
   314:     scorer. The true Pareto front and the metrics are computed host-side; this
   315:     process never sees them.
   316:     """
   317:     spec = _load_spec(env_key, seed)
   318:     n_var = int(spec["n_var"])
   319:     n_obj = int(spec["n_obj"])
   320:     bounds = tuple(spec["bounds"])
   321:     pop_size = int(spec["pop_size"])
   322:     n_gen = int(spec["n_gen"])
   323: 
   324:     # Black-box objective evaluator (legitimate: evaluating candidates is the task)
   325:     func = _build_objective(spec)
   326: 
   327:     # Set seeds
   328:     random.seed(seed)
   329:     np.random.seed(seed)
   330: 
   331:     # Determine individual class based on number of objectives
   332:     ind_class = creator.Individual3 if n_obj == 3 else creator.Individual
   333: 
   334:     # Initialize algorithm
   335:     moea = CustomMOEA(
   336:         pop_size=pop_size,
   337:         n_obj=n_obj,
   338:         n_var=n_var,
   339:         bounds=bounds,
   340:     )
   341: 
   342:     # Create initial population
   343:     population = [make_individual(n_var, bounds, ind_class) for _ in range(pop_size)]
   344: 
   345:     # Evaluate initial population
   346:     for ind in population:
   347:         ind.fitness.values = evaluate(ind, func)
   348: 
   349:     for gen in range(1, n_gen + 1):
   350:         # Parent selection
   351:         parents = moea.select(population, pop_size)
   352: 
   353:         # Variation (crossover + mutation)
   354:         offspring = moea.vary(parents)
   355: 
   356:         # Evaluate offspring
   357:         for ind in offspring:
   358:             if not ind.fitness.valid:
   359:                 ind.fitness.values = evaluate(ind, func)
   360: 
   361:         # Environmental selection (survival)
   362:         population = moea.survive(population, offspring)
   363: 
   364:         # Optional per-generation callback
   365:         moea.on_generation(gen, population)
   366: 
   367:         # Periodic progress feedback (objective-space extent only, no metrics)
   368:         if gen % 20 == 0 or gen == n_gen:
   369:             nd_front = get_nondominated(population)
   370:             front_values = np.array([ind.fitness.values for ind in nd_front])
   371:             print(
   372:                 f"TRAIN_PROGRESS gen={gen} front_size={len(nd_front)} "
   373:                 f"f_min={np.min(front_values, axis=0).round(4).tolist()} "
   374:                 f"f_max={np.max(front_values, axis=0).round(4).tolist()}",
   375:                 flush=True,
   376:             )
   377: 
   378:     # Final non-dominated front
   379:     nd_front = get_nondominated(population)
   380:     front_values = np.array([ind.fitness.values for ind in nd_front], dtype=np.float64)
   381: 
   382:     # Emit the final population's objective values for the host-side scorer. We do
   383:     # NOT have the true Pareto front, so we cannot (and do not) compute metrics.
   384:     payload = base64.b64encode(
   385:         np.ascontiguousarray(front_values, dtype=np.float64).tobytes()
   386:     ).decode("ascii")
   387:     print(
   388:         f"MOEA_PRED env={env_key} seed={seed} shape={front_values.shape[0]},{front_values.shape[1]} "
   389:         f"objs={payload}",
   390:         flush=True,
   391:     )
   392: 
   393:     # Save final front to disk (objective values only)
   394:     os.makedirs(output_dir, exist_ok=True)
   395:     np.savetxt(
   396:         os.path.join(output_dir, f"{env_key}_front.csv"),
   397:         front_values,
   398:         delimiter=",",
   399:         header=",".join(f"f{i+1}" for i in range(n_obj)),
   400:     )
   401: 
   402:     return front_values
   403: 
   404: 
   405: def main():
   406:     parser = argparse.ArgumentParser(description="Multi-Objective Optimization Benchmark")
   407:     parser.add_argument("--env", type=str, default=os.environ.get("ENV", ""))
   408:     parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", 42)))
   409:     parser.add_argument("--output-dir", type=str, default=os.environ.get("OUTPUT_DIR", "./output"))
   410:     args = parser.parse_args()
   411: 
   412:     if not args.env:
   413:         raise SystemExit("ENV not set")
   414: 
   415:     print(f"Running MOEA benchmark: {args.env} (seed={args.seed})", flush=True)
   416:     run_moea(args.env, args.seed, args.output_dir)
   417:     print(f"Done {args.env}.", flush=True)
   418: 
   419: 
   420: if __name__ == "__main__":
   421:     main()
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
Lines 159–221:
   156: # ================================================================
   157: 
   158: 
   159: 
   160: class CustomMOEA:
   161:     """NSGA-II: Non-dominated Sorting Genetic Algorithm II."""
   162: 
   163:     def __init__(self, pop_size, n_obj, n_var, bounds, cx_eta=20.0, mut_eta=20.0, mut_prob=None):
   164:         self.pop_size = pop_size
   165:         self.n_obj = n_obj
   166:         self.n_var = n_var
   167:         self.bounds = bounds
   168:         self.cx_eta = cx_eta
   169:         self.mut_eta = mut_eta
   170:         self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var
   171: 
   172:     def select(self, population, k):
   173:         """Binary tournament selection with crowding distance."""
   174:         fronts = tools.sortNondominated(population, len(population), first_front_only=False)
   175:         for front in fronts:
   176:             compute_crowding_distance(front)
   177:         return tools.selTournamentDCD(population, k)
   178: 
   179:     def vary(self, parents):
   180:         """SBX crossover + polynomial mutation."""
   181:         offspring = [deepcopy(ind) for ind in parents]
   182:         lo, hi = self.bounds
   183: 
   184:         for i in range(0, len(offspring) - 1, 2):
   185:             if random.random() < 0.9:
   186:                 tools.cxSimulatedBinaryBounded(
   187:                     offspring[i], offspring[i + 1],
   188:                     eta=self.cx_eta, low=lo, up=hi,
   189:                 )
   190:                 del offspring[i].fitness.values
   191:                 del offspring[i + 1].fitness.values
   192: 
   193:         for ind in offspring:
   194:             if random.random() < 1.0:
   195:                 tools.mutPolynomialBounded(
   196:                     ind, eta=self.mut_eta, low=lo, up=hi, indpb=self.mut_prob,
   197:                 )
   198:                 del ind.fitness.values
   199: 
   200:         return offspring
   201: 
   202:     def survive(self, population, offspring):
   203:         """NSGA-II survival: non-dominated sorting + crowding distance."""
   204:         combined = population + offspring
   205:         fronts = tools.sortNondominated(combined, self.pop_size, first_front_only=False)
   206: 
   207:         next_gen = []
   208:         for front in fronts:
   209:             if len(next_gen) + len(front) <= self.pop_size:
   210:                 next_gen.extend(front)
   211:             else:
   212:                 remaining = self.pop_size - len(next_gen)
   213:                 compute_crowding_distance(front)
   214:                 front.sort(key=lambda x: x.fitness.crowding_dist, reverse=True)
   215:                 next_gen.extend(front[:remaining])
   216:                 break
   217: 
   218:         return next_gen
   219: 
   220:     def on_generation(self, gen, population):
   221:         pass
   222: 
   223: 
   224: # ================================================================
```

### `moead` baseline — editable region  [READ-ONLY — reference implementation]

In `deap/custom_moea.py`:

```python
Lines 159–278:
   156: # ================================================================
   157: 
   158: 
   159: 
   160: class CustomMOEA:
   161:     """MOEA/D: Multi-Objective Evolutionary Algorithm Based on Decomposition."""
   162: 
   163:     def __init__(self, pop_size, n_obj, n_var, bounds, cx_eta=20.0, mut_eta=20.0, mut_prob=None):
   164:         self.pop_size = pop_size
   165:         self.n_obj = n_obj
   166:         self.n_var = n_var
   167:         self.bounds = bounds
   168:         self.cx_eta = cx_eta
   169:         self.mut_eta = mut_eta
   170:         self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var
   171:         self.T = 20  # neighborhood size
   172:         self.delta = 0.9  # probability of selecting from neighborhood
   173: 
   174:         # Generate weight vectors
   175:         self.weights = self._generate_weights(pop_size, n_obj)
   176:         self.pop_size = len(self.weights)  # adjust to actual number of weight vectors
   177: 
   178:         # Compute neighborhoods
   179:         self.neighbors = self._compute_neighborhoods()
   180: 
   181:         # Ideal point (updated during search)
   182:         self.z_star = None
   183: 
   184:     def _generate_weights(self, n, n_obj):
   185:         """Generate uniformly distributed weight vectors."""
   186:         if n_obj == 2:
   187:             weights = []
   188:             for i in range(n):
   189:                 w1 = i / max(n - 1, 1)
   190:                 weights.append([w1, 1.0 - w1])
   191:             return np.array(weights)
   192:         else:
   193:             # Use DEAP's uniform reference points for 3+ objectives
   194:             ref_points = tools.uniform_reference_points(n_obj, p=12)
   195:             return np.array(ref_points)
   196: 
   197:     def _compute_neighborhoods(self):
   198:         """Compute T-nearest weight vector neighborhoods."""
   199:         from scipy.spatial.distance import cdist
   200:         dist_matrix = cdist(self.weights, self.weights)
   201:         neighbors = []
   202:         for i in range(len(self.weights)):
   203:             idx = np.argsort(dist_matrix[i])[:self.T]
   204:             neighbors.append(idx.tolist())
   205:         return neighbors
   206: 
   207:     def _tchebycheff(self, fitness_values, weight, z_star):
   208:         """Tchebycheff scalarization."""
   209:         return max(weight[j] * abs(fitness_values[j] - z_star[j])
   210:                    for j in range(self.n_obj))
   211: 
   212:     def select(self, population, k):
   213:         """MOEA/D doesn't use standard selection — return population as-is."""
   214:         return [deepcopy(ind) for ind in population]
   215: 
   216:     def vary(self, parents):
   217:         """Generate one offspring per subproblem using neighborhood mating."""
   218:         offspring = []
   219:         lo, hi = self.bounds
   220: 
   221:         for i in range(len(parents)):
   222:             # Select mating pool (neighborhood or whole population)
   223:             if random.random() < self.delta:
   224:                 pool = [parents[j] for j in self.neighbors[i % len(self.neighbors)]]
   225:             else:
   226:                 pool = parents
   227: 
   228:             # Select two parents from pool
   229:             p1, p2 = random.sample(range(len(pool)), 2)
   230:             child = deepcopy(pool[p1])
   231: 
   232:             # SBX crossover
   233:             mate = deepcopy(pool[p2])
   234:             if random.random() < 1.0:
   235:                 tools.cxSimulatedBinaryBounded(child, mate, eta=self.cx_eta, low=lo, up=hi)
   236: 
   237:             # Polynomial mutation
   238:             tools.mutPolynomialBounded(child, eta=self.mut_eta, low=lo, up=hi, indpb=self.mut_prob)
   239:             del child.fitness.values
   240:             offspring.append(child)
   241: 
   242:         return offspring
   243: 
   244:     def survive(self, population, offspring):
   245:         """MOEA/D survival: update subproblems using Tchebycheff decomposition."""
   246:         # Update ideal point
   247:         all_inds = [ind for ind in population + offspring if ind.fitness.valid]
   248:         if not all_inds:
   249:             return population
   250: 
   251:         if self.z_star is None:
   252:             self.z_star = [float('inf')] * self.n_obj
   253:         for ind in all_inds:
   254:             for j in range(self.n_obj):
   255:                 if ind.fitness.values[j] < self.z_star[j]:
   256:                     self.z_star[j] = ind.fitness.values[j]
   257: 
   258:         # Update each subproblem
   259:         next_gen = list(population)
   260:         for i in range(min(len(offspring), len(self.weights))):
   261:             child = offspring[i]
   262:             if not child.fitness.valid:
   263:                 continue
   264: 
   265:             # Update neighbors
   266:             neighbors_idx = self.neighbors[i % len(self.neighbors)]
   267:             for j_idx in neighbors_idx:
   268:                 if j_idx >= len(next_gen):
   269:                     continue
   270:                 g_child = self._tchebycheff(child.fitness.values, self.weights[j_idx], self.z_star)
   271:                 g_current = self._tchebycheff(next_gen[j_idx].fitness.values, self.weights[j_idx], self.z_star)
   272:                 if g_child < g_current:
   273:                     next_gen[j_idx] = deepcopy(child)
   274: 
   275:         return next_gen[:self.pop_size]
   276: 
   277:     def on_generation(self, gen, population):
   278:         pass
   279: 
   280: 
   281: # ================================================================
```

### `spea2` baseline — editable region  [READ-ONLY — reference implementation]

In `deap/custom_moea.py`:

```python
Lines 159–227:
   156: # ================================================================
   157: 
   158: 
   159: 
   160: class CustomMOEA:
   161:     """SPEA2: Strength Pareto Evolutionary Algorithm 2."""
   162: 
   163:     def __init__(self, pop_size, n_obj, n_var, bounds, cx_eta=20.0, mut_eta=20.0, mut_prob=None):
   164:         self.pop_size = pop_size
   165:         self.n_obj = n_obj
   166:         self.n_var = n_var
   167:         self.bounds = bounds
   168:         self.cx_eta = cx_eta
   169:         self.mut_eta = mut_eta
   170:         self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var
   171:         self.archive = []
   172: 
   173:     def select(self, population, k):
   174:         """Binary tournament selection using SPEA2 fitness from archive."""
   175:         # Use archive for selection if available, otherwise population
   176:         pool = self.archive if self.archive else population
   177:         # Binary tournament on dominance
   178:         selected = []
   179:         for _ in range(k):
   180:             i1, i2 = random.sample(range(len(pool)), 2)
   181:             a, b = pool[i1], pool[i2]
   182:             if a.fitness.dominates(b.fitness):
   183:                 selected.append(deepcopy(a))
   184:             elif b.fitness.dominates(a.fitness):
   185:                 selected.append(deepcopy(b))
   186:             else:
   187:                 selected.append(deepcopy(random.choice([a, b])))
   188:         return selected
   189: 
   190:     def vary(self, parents):
   191:         """SBX crossover + polynomial mutation."""
   192:         offspring = [deepcopy(ind) for ind in parents]
   193:         lo, hi = self.bounds
   194: 
   195:         for i in range(0, len(offspring) - 1, 2):
   196:             if random.random() < 0.9:
   197:                 tools.cxSimulatedBinaryBounded(
   198:                     offspring[i], offspring[i + 1],
   199:                     eta=self.cx_eta, low=lo, up=hi,
   200:                 )
   201:                 del offspring[i].fitness.values
   202:                 del offspring[i + 1].fitness.values
   203: 
   204:         for ind in offspring:
   205:             if random.random() < 1.0:
   206:                 tools.mutPolynomialBounded(
   207:                     ind, eta=self.mut_eta, low=lo, up=hi, indpb=self.mut_prob,
   208:                 )
   209:                 del ind.fitness.values
   210: 
   211:         return offspring
   212: 
   213:     def survive(self, population, offspring):
   214:         """SPEA2 survival: strength fitness + kNN density truncation."""
   215:         combined = population + offspring
   216: 
   217:         # Use DEAP's built-in SPEA2 selection
   218:         selected = tools.selSPEA2(combined, self.pop_size)
   219: 
   220:         # Update archive with non-dominated solutions
   221:         nd = get_nondominated(selected)
   222:         self.archive = [deepcopy(ind) for ind in nd[:self.pop_size]]
   223: 
   224:         return selected
   225: 
   226:     def on_generation(self, gen, population):
   227:         pass
   228: 
   229: 
   230: # ================================================================
```

### `nsga3` baseline — editable region  [READ-ONLY — reference implementation]

In `deap/custom_moea.py`:

```python
Lines 159–217:
   156: # ================================================================
   157: 
   158: 
   159: 
   160: class CustomMOEA:
   161:     """NSGA-III: Non-dominated Sorting Genetic Algorithm III."""
   162: 
   163:     def __init__(self, pop_size, n_obj, n_var, bounds, cx_eta=20.0, mut_eta=20.0, mut_prob=None):
   164:         self.pop_size = pop_size
   165:         self.n_obj = n_obj
   166:         self.n_var = n_var
   167:         self.bounds = bounds
   168:         self.cx_eta = cx_eta
   169:         self.mut_eta = mut_eta
   170:         self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var
   171: 
   172:         # Generate reference points
   173:         if n_obj == 2:
   174:             p = pop_size - 1  # number of divisions
   175:             self.ref_points = tools.uniform_reference_points(n_obj, p=p)
   176:         else:
   177:             self.ref_points = tools.uniform_reference_points(n_obj, p=12)
   178: 
   179:     def select(self, population, k):
   180:         """Random shuffle selection (NSGA-III relies on survive for diversity)."""
   181:         selected = [deepcopy(ind) for ind in population]
   182:         random.shuffle(selected)
   183:         return selected[:k]
   184: 
   185:     def vary(self, parents):
   186:         """SBX crossover + polynomial mutation."""
   187:         offspring = [deepcopy(ind) for ind in parents]
   188:         lo, hi = self.bounds
   189: 
   190:         for i in range(0, len(offspring) - 1, 2):
   191:             if random.random() < 1.0:
   192:                 tools.cxSimulatedBinaryBounded(
   193:                     offspring[i], offspring[i + 1],
   194:                     eta=self.cx_eta, low=lo, up=hi,
   195:                 )
   196:                 del offspring[i].fitness.values
   197:                 del offspring[i + 1].fitness.values
   198: 
   199:         for ind in offspring:
   200:             if random.random() < 1.0:
   201:                 tools.mutPolynomialBounded(
   202:                     ind, eta=self.mut_eta, low=lo, up=hi, indpb=self.mut_prob,
   203:                 )
   204:                 del ind.fitness.values
   205: 
   206:         return offspring
   207: 
   208:     def survive(self, population, offspring):
   209:         """NSGA-III survival: reference-point-based selection."""
   210:         combined = population + offspring
   211: 
   212:         # Use DEAP's built-in NSGA-III selection
   213:         selected = tools.selNSGA3(combined, self.pop_size, self.ref_points)
   214:         return selected
   215: 
   216:     def on_generation(self, gen, population):
   217:         pass
   218: 
   219: 
   220: # ================================================================
```

### `rvea` baseline — editable region  [READ-ONLY — reference implementation]

In `deap/custom_moea.py`:

```python
Lines 159–309:
   156: # ================================================================
   157: 
   158: 
   159: 
   160: class CustomMOEA:
   161:     """RVEA: Reference Vector Guided Evolutionary Algorithm."""
   162: 
   163:     def __init__(self, pop_size, n_obj, n_var, bounds, cx_eta=20.0, mut_eta=20.0, mut_prob=None):
   164:         self.pop_size = pop_size
   165:         self.n_obj = n_obj
   166:         self.n_var = n_var
   167:         self.bounds = bounds
   168:         self.cx_eta = cx_eta
   169:         self.mut_eta = mut_eta
   170:         self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var
   171:         self.alpha = 2.0  # penalty parameter for APD
   172:         self.fr = 0.1  # frequency of reference vector adaptation
   173: 
   174:         # Generate initial reference vectors
   175:         if n_obj == 2:
   176:             p = pop_size - 1
   177:             self.ref_vectors = np.array(tools.uniform_reference_points(n_obj, p=p))
   178:         else:
   179:             self.ref_vectors = np.array(tools.uniform_reference_points(n_obj, p=12))
   180:         self.ref_vectors_initial = self.ref_vectors.copy()
   181: 
   182:         # Normalize reference vectors to unit length
   183:         norms = np.linalg.norm(self.ref_vectors, axis=1, keepdims=True)
   184:         norms[norms < 1e-12] = 1e-12
   185:         self.ref_vectors = self.ref_vectors / norms
   186: 
   187:     def _angle_penalized_distance(self, fitness_values, gen, max_gen):
   188:         """Compute angle-penalized distance for each individual to its closest reference vector."""
   189:         F = np.array(fitness_values)
   190:         n = len(F)
   191:         n_ref = len(self.ref_vectors)
   192: 
   193:         if n == 0:
   194:             return np.array([]), np.array([])
   195: 
   196:         # Translate objectives (subtract ideal point)
   197:         z_min = np.min(F, axis=0)
   198:         F_translated = F - z_min + 1e-12
   199: 
   200:         # Compute angles between each individual and each reference vector
   201:         # cos(theta) = (F . v) / (||F|| * ||v||)
   202:         F_norms = np.linalg.norm(F_translated, axis=1, keepdims=True)
   203:         F_norms[F_norms < 1e-12] = 1e-12
   204:         F_normalized = F_translated / F_norms
   205: 
   206:         # Cosine similarity
   207:         cos_angles = F_normalized @ self.ref_vectors.T  # (n, n_ref)
   208:         cos_angles = np.clip(cos_angles, -1.0, 1.0)
   209:         angles = np.arccos(cos_angles)  # (n, n_ref)
   210: 
   211:         # Associate each individual with closest reference vector
   212:         associations = np.argmin(angles, axis=1)  # (n,)
   213:         min_angles = angles[np.arange(n), associations]  # (n,)
   214: 
   215:         # Compute convergence (distance along reference vector)
   216:         convergence = F_norms.flatten()
   217: 
   218:         # Angle penalty that increases over generations
   219:         gamma = self.alpha * (gen / max(max_gen, 1)) ** 2
   220: 
   221:         # APD = convergence * (1 + gamma * angle)
   222:         apd = convergence * (1.0 + gamma * min_angles)
   223: 
   224:         return apd, associations
   225: 
   226:     def select(self, population, k):
   227:         """Random mating selection."""
   228:         selected = [deepcopy(ind) for ind in population]
   229:         random.shuffle(selected)
   230:         return selected[:k]
   231: 
   232:     def vary(self, parents):
   233:         """SBX crossover + polynomial mutation."""
   234:         offspring = [deepcopy(ind) for ind in parents]
   235:         lo, hi = self.bounds
   236: 
   237:         for i in range(0, len(offspring) - 1, 2):
   238:             if random.random() < 1.0:
   239:                 tools.cxSimulatedBinaryBounded(
   240:                     offspring[i], offspring[i + 1],
   241:                     eta=self.cx_eta, low=lo, up=hi,
   242:                 )
   243:                 del offspring[i].fitness.values
   244:                 del offspring[i + 1].fitness.values
   245: 
   246:         for ind in offspring:
   247:             if random.random() < 1.0:
   248:                 tools.mutPolynomialBounded(
   249:                     ind, eta=self.mut_eta, low=lo, up=hi, indpb=self.mut_prob,
   250:                 )
   251:                 del ind.fitness.values
   252: 
   253:         return offspring
   254: 
   255:     def survive(self, population, offspring):
   256:         """RVEA survival: angle-penalized distance based selection."""
   257:         combined = population + offspring
   258:         valid = [ind for ind in combined if ind.fitness.valid]
   259: 
   260:         if len(valid) <= self.pop_size:
   261:             return valid
   262: 
   263:         fitness_values = [ind.fitness.values for ind in valid]
   264:         # Use a large gen estimate based on problem config
   265:         max_gen = 400
   266:         gen_estimate = getattr(self, '_current_gen', max_gen // 2)
   267:         apd, associations = self._angle_penalized_distance(fitness_values, gen_estimate, max_gen)
   268: 
   269:         # Select the best individual per reference vector (lowest APD)
   270:         selected_indices = set()
   271:         n_ref = len(self.ref_vectors)
   272:         for v in range(n_ref):
   273:             mask = np.where(associations == v)[0]
   274:             if len(mask) > 0:
   275:                 best_idx = mask[np.argmin(apd[mask])]
   276:                 selected_indices.add(best_idx)
   277: 
   278:         # If not enough, fill with best remaining by APD
   279:         if len(selected_indices) < self.pop_size:
   280:             remaining = [i for i in range(len(valid)) if i not in selected_indices]
   281:             remaining.sort(key=lambda i: apd[i])
   282:             for i in remaining:
   283:                 selected_indices.add(i)
   284:                 if len(selected_indices) >= self.pop_size:
   285:                     break
   286: 
   287:         # If too many (more ref vectors than pop_size), truncate by APD
   288:         selected_list = sorted(selected_indices, key=lambda i: apd[i])[:self.pop_size]
   289:         return [valid[i] for i in selected_list]
   290: 
   291:     def on_generation(self, gen, population):
   292:         """Adapt reference vectors periodically."""
   293:         self._current_gen = gen
   294: 
   295:         # Reference vector adaptation
   296:         max_gen = 400
   297:         if gen % max(1, int(self.fr * max_gen)) == 0 and len(population) > 0:
   298:             fitness_values = np.array([ind.fitness.values for ind in population if ind.fitness.valid])
   299:             if len(fitness_values) > 0:
   300:                 z_max = np.max(fitness_values, axis=0)
   301:                 z_min = np.min(fitness_values, axis=0)
   302:                 scale = z_max - z_min
   303:                 scale[scale < 1e-12] = 1.0
   304: 
   305:                 # Scale reference vectors
   306:                 self.ref_vectors = self.ref_vectors_initial * scale
   307:                 norms = np.linalg.norm(self.ref_vectors, axis=1, keepdims=True)
   308:                 norms[norms < 1e-12] = 1e-12
   309:                 self.ref_vectors = self.ref_vectors / norms
   310: 
   311: 
   312: # ================================================================
```

### `agemoea` baseline — editable region  [READ-ONLY — reference implementation]

In `deap/custom_moea.py`:

```python
Lines 159–333:
   156: # ================================================================
   157: 
   158: 
   159: 
   160: class CustomMOEA:
   161:     """AGE-MOEA: Adaptive Geometry Estimation based MOEA."""
   162: 
   163:     def __init__(self, pop_size, n_obj, n_var, bounds, cx_eta=20.0, mut_eta=20.0, mut_prob=None):
   164:         self.pop_size = pop_size
   165:         self.n_obj = n_obj
   166:         self.n_var = n_var
   167:         self.bounds = bounds
   168:         self.cx_eta = cx_eta
   169:         self.mut_eta = mut_eta
   170:         self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var
   171: 
   172:     def _estimate_geometry(self, front_values):
   173:         """Estimate the geometry parameter p of the Pareto front.
   174: 
   175:         Uses the relationship between Lp-norm and front shape:
   176:         p=1: linear front (like DTLZ1)
   177:         p=2: spherical front (like DTLZ2)
   178:         p->inf: rectangular front
   179:         """
   180:         if len(front_values) < 2 or self.n_obj < 2:
   181:             return 1.0
   182: 
   183:         F = np.array(front_values)
   184: 
   185:         # Normalize objectives
   186:         z_min = np.min(F, axis=0)
   187:         z_max = np.max(F, axis=0)
   188:         scale = z_max - z_min
   189:         scale[scale < 1e-12] = 1.0
   190:         F_norm = (F - z_min) / scale
   191: 
   192:         # Find extreme points (closest to axes)
   193:         extremes = []
   194:         for m in range(self.n_obj):
   195:             # Point with smallest value on objective m
   196:             idx = np.argmin(F_norm[:, m])
   197:             extremes.append(F_norm[idx])
   198: 
   199:         if len(extremes) < 2:
   200:             return 1.0
   201: 
   202:         # Estimate p from extreme points
   203:         # For an Lp-norm sphere of radius r: sum(|x_i/r|^p) = 1
   204:         # Use the median point on the front to estimate p
   205:         median_idx = len(F_norm) // 2
   206:         median_point = np.sort(F_norm, axis=0)[median_idx]
   207: 
   208:         # Avoid zero/negative values
   209:         median_point = np.maximum(median_point, 1e-8)
   210: 
   211:         # Binary search for p
   212:         p_low, p_high = 0.1, 20.0
   213:         for _ in range(50):
   214:             p_mid = (p_low + p_high) / 2
   215:             lp_val = np.sum(median_point ** p_mid)
   216:             if lp_val > 1.0:
   217:                 p_low = p_mid
   218:             else:
   219:                 p_high = p_mid
   220:         p = (p_low + p_high) / 2
   221:         return max(0.1, min(p, 20.0))
   222: 
   223:     def _survival_score(self, front_values, p):
   224:         """Compute survival score based on Lp-distance-based crowding."""
   225:         F = np.array(front_values)
   226:         n = len(F)
   227:         if n <= 2:
   228:             return np.full(n, float('inf'))
   229: 
   230:         # Normalize
   231:         z_min = np.min(F, axis=0)
   232:         z_max = np.max(F, axis=0)
   233:         scale = z_max - z_min
   234:         scale[scale < 1e-12] = 1.0
   235:         F_norm = (F - z_min) / scale
   236: 
   237:         # Compute pairwise Lp-distances
   238:         scores = np.zeros(n)
   239:         for i in range(n):
   240:             dists = []
   241:             for j in range(n):
   242:                 if i == j:
   243:                     continue
   244:                 diff = np.abs(F_norm[i] - F_norm[j])
   245:                 lp_dist = np.sum(diff ** p) ** (1.0 / p)
   246:                 dists.append(lp_dist)
   247:             dists.sort()
   248:             # Use nearest neighbor distance as diversity score
   249:             if dists:
   250:                 scores[i] = dists[0]
   251:             else:
   252:                 scores[i] = 0.0
   253: 
   254:         return scores
   255: 
   256:     def select(self, population, k):
   257:         """Binary tournament selection based on non-domination rank."""
   258:         fronts = tools.sortNondominated(population, len(population), first_front_only=False)
   259:         # Assign rank
   260:         for rank, front in enumerate(fronts):
   261:             for ind in front:
   262:                 ind.fitness.crowding_dist = 0.0  # reset
   263:                 ind._rank = rank
   264:         # Tournament
   265:         selected = []
   266:         for _ in range(k):
   267:             i1, i2 = random.sample(range(len(population)), 2)
   268:             a, b = population[i1], population[i2]
   269:             if a._rank < b._rank:
   270:                 selected.append(deepcopy(a))
   271:             elif b._rank < a._rank:
   272:                 selected.append(deepcopy(b))
   273:             else:
   274:                 selected.append(deepcopy(random.choice([a, b])))
   275:         return selected
   276: 
   277:     def vary(self, parents):
   278:         """SBX crossover + polynomial mutation."""
   279:         offspring = [deepcopy(ind) for ind in parents]
   280:         lo, hi = self.bounds
   281: 
   282:         for i in range(0, len(offspring) - 1, 2):
   283:             if random.random() < 0.9:
   284:                 tools.cxSimulatedBinaryBounded(
   285:                     offspring[i], offspring[i + 1],
   286:                     eta=self.cx_eta, low=lo, up=hi,
   287:                 )
   288:                 del offspring[i].fitness.values
   289:                 del offspring[i + 1].fitness.values
   290: 
   291:         for ind in offspring:
   292:             if random.random() < 1.0:
   293:                 tools.mutPolynomialBounded(
   294:                     ind, eta=self.mut_eta, low=lo, up=hi, indpb=self.mut_prob,
   295:                 )
   296:                 del ind.fitness.values
   297: 
   298:         return offspring
   299: 
   300:     def survive(self, population, offspring):
   301:         """AGE-MOEA survival: adaptive geometry-based selection."""
   302:         combined = population + offspring
   303: 
   304:         # Non-dominated sorting
   305:         fronts = tools.sortNondominated(combined, len(combined), first_front_only=False)
   306: 
   307:         next_gen = []
   308:         for front_idx, front in enumerate(fronts):
   309:             if len(next_gen) + len(front) <= self.pop_size:
   310:                 next_gen.extend(front)
   311:             else:
   312:                 remaining = self.pop_size - len(next_gen)
   313:                 if remaining <= 0:
   314:                     break
   315: 
   316:                 # Estimate geometry from the first front
   317:                 first_front_values = [ind.fitness.values for ind in fronts[0]]
   318:                 p = self._estimate_geometry(first_front_values)
   319: 
   320:                 # Compute survival scores for the critical front
   321:                 front_values = [ind.fitness.values for ind in front]
   322:                 scores = self._survival_score(front_values, p)
   323: 
   324:                 # Select individuals with highest diversity scores
   325:                 sorted_indices = np.argsort(-scores)  # descending
   326:                 for idx in sorted_indices[:remaining]:
   327:                     next_gen.append(front[idx])
   328:                 break
   329: 
   330:         return next_gen
   331: 
   332:     def on_generation(self, gen, population):
   333:         pass
   334: 
   335: 
   336: # ================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

## Time Budget

You have **5 hours** of wall-clock time before submission, covering
everything you do here: reading the code, editing it, and any trial runs
you launch.

Good luck.
