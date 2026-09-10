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

## Evaluation
Strategies are evaluated on benchmarks (all minimization, lower is better):

| Benchmark | Function | Dimensions | Domain | Global Minimum |
|-----------|----------|------------|--------|----------------|
| rastrigin-30d | Rastrigin | 30 | [-5.12, 5.12] | 0 |
| rosenbrock-30d | Rosenbrock | 30 | [-5, 10] | 0 |
| ackley-30d | Ackley | 30 | [-32.768, 32.768] | 0 |
| rastrigin-100d | Rastrigin | 100 | [-5.12, 5.12] | 0 |

**Metrics**: `best_fitness` (final best value, lower is better) and `convergence_gen` (generation reaching near-final fitness).

## Baselines (paper-cited reference implementations)
- **ga_sbx** — Genetic Algorithm with Simulated Binary Crossover and Polynomial Mutation (Deb and Agrawal, 1995); paper-default `eta_c = eta_m = 20`, mutation probability `1/n`.
- **de** — Classical DE/rand/1/bin (Storn and Price, 1997); paper-default `F = 0.5`, `CR = 0.9`.
- **lshade** — L-SHADE (Tanabe and Fukunaga, IEEE CEC 2014); paper-default initial population `18 * n`, archive size `2.6 * pop`, history memory `H = 6`, linear population reduction to `N_min = 4`.
