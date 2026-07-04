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

## Evaluation
Your strategy is evaluated on a set of held-out continuous multi-objective
benchmark problems drawn from the standard ZDT and DTLZ families (a mix of
2- and 3-objective problems with convex, disconnected, and spherical/linear
Pareto fronts), run with multiple seeds. The harness supplies your `CustomMOEA`
with the objective functions to optimize and the problem configuration
(population size, number of generations, variables, and bounds) at run time; the
specific problem identity and its true Pareto front are held out and are not
available to your program. Your strategy receives individuals whose objective
values have already been evaluated — design it to work for both 2- and
3-objective problems.

Three metrics are reported per problem:
- **Hypervolume (HV)**: volume of objective space dominated by the Pareto front approximation. **Higher is better.**
- **Inverted Generational Distance (IGD)**: average distance from true Pareto front points to nearest found solution. **Lower is better.**
- **Spread**: uniformity of the Pareto front approximation. **Lower is better.**

## Baselines (paper-cited reference implementations)
- **nsga2** — Deb et al. (*IEEE TEC* 2002); paper-default SBX `eta_c = 20`, polynomial mutation `eta_m = 20`, `p_m = 1/n_var`.
- **moead** — Zhang and Li (*IEEE TEC* 2007); paper-default Tchebycheff aggregation, neighborhood size `T = 20`.
- **spea2** — Zitzler, Laumanns, and Thiele (EUROGEN 2001 / TIK-Report 103); paper-default archive size = population size, `k = sqrt(N + |archive|)` for k-NN density.
- **nsga3** — Deb and Jain (*IEEE TEC* 2014); paper-default Das–Dennis reference points with divisions chosen from objective dimensionality.
- **rvea** — Cheng, Jin, Olhofer, and Sendhoff (*IEEE TEC* 2016); paper-default angle-penalized distance with `alpha = 2`, reference-vector adaptation period `fr = 0.1`.
- **agemoea** — Panichella (GECCO 2019); paper-default geometry-estimated Minkowski-`p` survival.
