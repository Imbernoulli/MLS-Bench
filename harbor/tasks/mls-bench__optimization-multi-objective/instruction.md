# MLS-Bench: optimization-multi-objective

# Multi-Objective Optimization: Custom Evolutionary Strategy Design

## Research Question
Design a novel multi-objective evolutionary algorithm (MOEA) strategy that achieves strong convergence, diversity, and spread on multi-objective optimization problems.

## Background
Multi-objective optimization aims to find a set of Pareto-optimal solutions that represent the best trade-offs among conflicting objectives. Evolutionary algorithms are the dominant approach, differing primarily in three components:

- **Parent selection**: how to choose individuals for mating.
- **Variation**: how to produce offspring via crossover and mutation operators.
- **Environmental selection (survival)**: how to prune the combined parent + offspring pool back to population size.

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
- `tools.selNSGA3(pop, k, ref_points)` -> reference-point-based selection.
- `tools.cxSimulatedBinaryBounded(ind1, ind2, eta, low, up)` -> SBX crossover.
- `tools.mutPolynomialBounded(ind, eta, low, up, indpb)` -> polynomial mutation.
- `tools.uniform_reference_points(nobj, p)` -> generate reference points.
- `compute_crowding_distance(individuals)` -> sets `.fitness.crowding_dist`.
- `get_nondominated(population)` -> first non-dominated front.

## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/deap/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be rejected.

- `deap/custom_moea.py`
- editable lines **297–441**

## Readable Context

### `deap/custom_moea.py`  [EDITABLE — lines 297–441 only]

```python
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
   350:         and crowding distance. Replace with a better strategy.
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
   401:         Default: non-dominated sorting + crowding distance.
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
```

## Tips

- Keep the function/class signatures of the editable regions identical;
  the editable region is imported by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.

Good luck.
