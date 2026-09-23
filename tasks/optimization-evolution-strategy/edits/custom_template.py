#!/usr/bin/env python3
"""Evolutionary Optimization Strategy Benchmark.

This script benchmarks an evolutionary optimization strategy on standard
continuous optimization test functions (Rastrigin, Rosenbrock, Ackley).
The goal is to minimize each function by designing effective selection,
crossover, and mutation operators.

Usage:
    python deap/custom_evolution.py --function rastrigin --dim 30 --seed 42
"""

import argparse
import math
import random
import time
from typing import List, Tuple, Callable

import numpy as np
from deap import base, creator, tools

# ================================================================
# FIXED — Benchmark functions and infrastructure (do not modify)
# ================================================================

# --- Benchmark function definitions ---

def rastrigin(individual: List[float]) -> Tuple[float]:
    """Rastrigin function. Global minimum: f(0,...,0) = 0. Domain: [-5.12, 5.12]."""
    A = 10.0
    n = len(individual)
    val = A * n + sum(x**2 - A * math.cos(2 * math.pi * x) for x in individual)
    return (val,)


def rosenbrock(individual: List[float]) -> Tuple[float]:
    """Rosenbrock function. Global minimum: f(1,...,1) = 0. Domain: [-5, 10]."""
    val = sum(
        100.0 * (individual[i + 1] - individual[i]**2)**2 + (1 - individual[i])**2
        for i in range(len(individual) - 1)
    )
    return (val,)


def ackley(individual: List[float]) -> Tuple[float]:
    """Ackley function. Global minimum: f(0,...,0) = 0. Domain: [-32.768, 32.768]."""
    n = len(individual)
    sum_sq = sum(x**2 for x in individual) / n
    sum_cos = sum(math.cos(2 * math.pi * x) for x in individual) / n
    val = -20.0 * math.exp(-0.2 * math.sqrt(sum_sq)) - math.exp(sum_cos) + 20.0 + math.e
    return (val,)


BENCHMARKS = {
    "rastrigin": {"func": rastrigin, "bounds": (-5.12, 5.12)},
    "rosenbrock": {"func": rosenbrock, "bounds": (-5.0, 10.0)},
    "ackley": {"func": ackley, "bounds": (-32.768, 32.768)},
}


# --- DEAP fitness and individual setup ---

# Single-objective minimization
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)


def make_individual(toolbox, dim: int, lo: float, hi: float):
    """Create a random individual within bounds."""
    ind = creator.Individual([random.uniform(lo, hi) for _ in range(dim)])
    return ind


def clip_individual(individual, lo: float, hi: float):
    """Clip individual's genes to stay within bounds."""
    for i in range(len(individual)):
        individual[i] = max(lo, min(hi, individual[i]))
    return individual


# ================================================================
# EDITABLE SECTION — Design your evolutionary strategy below
# (lines 87 to 225)
# ================================================================

def custom_select(population: list, k: int, toolbox=None) -> list:
    """Select k individuals from the population.

    Args:
        population: List of individuals (each has a .fitness.values attribute).
        k: Number of individuals to select.
        toolbox: The DEAP toolbox (optional, for access to other operators).

    Returns:
        List of k selected individuals (deep copies recommended).
    """
    # Default: tournament selection with tournament size 3
    return tools.selTournament(population, k, tournsize=3)


def custom_crossover(ind1: list, ind2: list) -> Tuple[list, list]:
    """Apply crossover to two individuals.

    Args:
        ind1, ind2: Parent individuals (lists of floats).

    Returns:
        Tuple of two offspring individuals (modified in-place).
    """
    # Default: simulated binary crossover (SBX), eta=20
    tools.cxSimulatedBinary(ind1, ind2, eta=20.0)
    return ind1, ind2


def custom_mutate(individual: list, lo: float, hi: float) -> Tuple[list]:
    """Apply mutation to an individual.

    Args:
        individual: The individual to mutate (list of floats).
        lo: Lower bound for genes.
        hi: Upper bound for genes.

    Returns:
        Tuple containing the mutated individual.
    """
    # Default: polynomial mutation, eta=20, indpb=1/dim
    tools.mutPolynomialBounded(
        individual, eta=20.0, low=lo, up=hi,
        indpb=1.0 / len(individual)
    )
    return (individual,)


def run_evolution(
    evaluate_func: Callable,
    dim: int,
    lo: float,
    hi: float,
    pop_size: int,
    n_generations: int,
    cx_prob: float,
    mut_prob: float,
    seed: int,
) -> Tuple[list, list]:
    """Run the evolutionary algorithm.

    Args:
        evaluate_func: Fitness function mapping individual -> (fitness_value,).
        dim: Dimensionality of the search space.
        lo: Lower bound for each dimension.
        hi: Upper bound for each dimension.
        pop_size: Population size.
        n_generations: Number of generations.
        cx_prob: Crossover probability.
        mut_prob: Mutation probability.
        seed: Random seed.

    Returns:
        best_individual: The best individual found.
        fitness_history: List of best fitness per generation.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Setup toolbox
    toolbox = base.Toolbox()
    toolbox.register("individual", make_individual, toolbox, dim, lo, hi)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_func)

    # Initialize population
    pop = toolbox.population(n=pop_size)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    fitness_history = []

    for gen in range(n_generations):
        # Selection
        offspring = custom_select(pop, len(pop), toolbox)
        offspring = [toolbox.clone(ind) for ind in offspring]

        # Crossover
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < cx_prob:
                custom_crossover(offspring[i], offspring[i + 1])
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values

        # Mutation
        for i in range(len(offspring)):
            if random.random() < mut_prob:
                custom_mutate(offspring[i], lo, hi)
                del offspring[i].fitness.values

        # Clip to bounds
        for ind in offspring:
            clip_individual(ind, lo, hi)

        # Evaluate individuals with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace population
        pop[:] = offspring

        # Track best fitness
        best_fit = min(ind.fitness.values[0] for ind in pop)
        fitness_history.append(best_fit)

        if (gen + 1) % 50 == 0 or gen == 0:
            avg_fit = sum(ind.fitness.values[0] for ind in pop) / len(pop)
            print(
                f"TRAIN_METRICS gen={gen+1} best_fitness={best_fit:.6e} "
                f"avg_fitness={avg_fit:.6e}",
                flush=True,
            )

    best_ind = min(pop, key=lambda ind: ind.fitness.values[0])
    return best_ind, fitness_history

# ================================================================
# FIXED — Evaluation harness (do not modify below)
# ================================================================


def _fresh_math():
    """A private instance of the math module, immune to monkeypatching of `math`."""
    import importlib.util
    spec = importlib.util.find_spec("math")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_M = _fresh_math()


# Reference objectives for scoring: the same formulas as the benchmark
# functions at the top of this file, bound here (after the editable section)
# so that nothing the editable code does to `rastrigin`, `BENCHMARKS`,
# `math`, or the returned individual's `.fitness` can change the score.
def _ref_rastrigin(x):
    A = 10.0
    n = len(x)
    return A * n + sum(v**2 - A * _M.cos(2 * _M.pi * v) for v in x)


def _ref_rosenbrock(x):
    return sum(
        100.0 * (x[i + 1] - x[i]**2)**2 + (1 - x[i])**2
        for i in range(len(x) - 1)
    )


def _ref_ackley(x):
    n = len(x)
    sum_sq = sum(v**2 for v in x) / n
    sum_cos = sum(_M.cos(2 * _M.pi * v) for v in x) / n
    return -20.0 * _M.exp(-0.2 * _M.sqrt(sum_sq)) - _M.exp(sum_cos) + 20.0 + _M.e


_REFERENCE = {
    "rastrigin": (_ref_rastrigin, (-5.12, 5.12)),
    "rosenbrock": (_ref_rosenbrock, (-5.0, 10.0)),
    "ackley": (_ref_ackley, (-32.768, 32.768)),
}


def _score_returned(best_ind, name: str, dim: int) -> float:
    """Re-evaluate the returned individual with the reference objective.

    The reported best_fitness is the true objective value at the returned
    point, never the individual's self-reported `.fitness`. A point with the
    wrong dimension, a non-finite coordinate, or a coordinate outside the
    domain is rejected (no TEST_METRICS line is printed).
    """
    func, (lo, hi) = _REFERENCE[name]
    try:
        x = [float(v) for v in (list.__iter__(best_ind) if isinstance(best_ind, list) else best_ind)]
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"ERROR: returned best_individual is not a vector of floats: {exc}")
    if len(x) != dim:
        raise SystemExit(f"ERROR: returned best_individual has {len(x)} genes, expected {dim}")
    bad = [i for i, v in enumerate(x) if not (_M.isfinite(v) and lo <= v <= hi)]
    if bad:
        raise SystemExit(
            f"ERROR: returned best_individual violates the domain [{lo}, {hi}] "
            f"at {len(bad)} coordinate(s), e.g. index {bad[0]} = {x[bad[0]]!r}"
        )
    val = float(func(x))
    if not _M.isfinite(val):
        raise SystemExit(f"ERROR: objective at returned best_individual is not finite: {val!r}")
    return val


def compute_convergence_gen(fitness_history: list, threshold_ratio: float = 0.01) -> int:
    """Compute the generation at which fitness first reaches within threshold of final best.

    Returns the 1-indexed generation number, or len(fitness_history) if never converged.
    """
    if not fitness_history:
        return 0
    final_best = fitness_history[-1]
    # threshold: within 1% of final best, or absolute 1e-6 for near-zero
    threshold = max(abs(final_best) * threshold_ratio, 1e-6)
    for i, f in enumerate(fitness_history):
        if abs(f - final_best) <= threshold:
            return i + 1
    return len(fitness_history)


def main():
    parser = argparse.ArgumentParser(description="Evolutionary Optimization Benchmark")
    parser.add_argument("--function", type=str, required=True,
                        choices=list(_REFERENCE.keys()),
                        help="Benchmark function to optimize")
    parser.add_argument("--dim", type=int, default=30,
                        help="Dimensionality of the search space (default: 30)")
    parser.add_argument("--pop-size", type=int, default=200,
                        help="Population size (default: 200)")
    parser.add_argument("--n-generations", type=int, default=500,
                        help="Number of generations (default: 500)")
    parser.add_argument("--cx-prob", type=float, default=0.9,
                        help="Crossover probability (default: 0.9)")
    parser.add_argument("--mut-prob", type=float, default=0.2,
                        help="Mutation probability (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    true_func, (lo, hi) = _REFERENCE[args.function]
    n_evals = [0]

    def evaluate_func(individual):
        n_evals[0] += 1
        return (true_func(individual),)

    evaluate_func.__name__ = args.function

    print(f"=== {args.function.upper()} (dim={args.dim}) ===", flush=True)
    print(f"Bounds: [{lo}, {hi}], Pop: {args.pop_size}, Gens: {args.n_generations}", flush=True)

    t0 = time.time()
    best_ind, fitness_history = run_evolution(
        evaluate_func=evaluate_func,
        dim=args.dim,
        lo=lo,
        hi=hi,
        pop_size=args.pop_size,
        n_generations=args.n_generations,
        cx_prob=args.cx_prob,
        mut_prob=args.mut_prob,
        seed=args.seed,
    )
    elapsed = time.time() - t0

    best_fitness = _score_returned(best_ind, args.function, args.dim)
    try:
        claimed = float(best_ind.fitness.values[0])
    except Exception:  # noqa: BLE001
        claimed = float("nan")
    if not (abs(claimed - best_fitness) <= 1e-9 * max(1.0, abs(best_fitness))):
        print(f"WARNING: self-reported best fitness {claimed!r} differs from the "
              f"re-evaluated value {best_fitness!r}; the re-evaluated value is reported.",
              flush=True)
    convergence_gen = compute_convergence_gen(fitness_history)

    print(f"\n=== Results ===", flush=True)
    print(f"Best fitness: {best_fitness:.6e}", flush=True)
    print(f"Convergence generation: {convergence_gen}/{args.n_generations}", flush=True)
    print(f"Function evaluations: {n_evals[0]}", flush=True)
    print(f"Wall time: {elapsed:.1f}s", flush=True)
    print(
        f"TEST_METRICS best_fitness={best_fitness:.6e} "
        f"convergence_gen={convergence_gen}",
        flush=True,
    )


if __name__ == "__main__":
    main()
