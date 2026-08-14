"""
Multi-Objective Optimization — Custom Evolutionary Strategy Template

This script runs a complete multi-objective evolutionary algorithm on a held-out
benchmark problem. The agent should implement the custom selection and variation
strategy in the CustomMOEA class.

NOTE: The benchmark problem identity, its analytic true Pareto front, and the
evaluation metrics are NOT part of this program. The harness pre-generates the
problem to optimize (the objective functions, with their numeric configuration)
and scores the final population in a separate host-side process. Your strategy
only ever sees individuals with already-evaluated objective values — it never
receives the problem name nor the true front.

Usage (the harness sets ENV/SEED for you):
    ENV=<opaque-problem-key> SEED=42 python deap/custom_moea.py
"""

import argparse
import base64
import io
import json
import marshal
import math
import os
import random
import time
import types
import warnings
from copy import deepcopy
from functools import reduce
from math import cos, pi, sin, sqrt
from operator import mul
from typing import List, Optional, Tuple

import numpy as np

from deap import base, benchmarks, creator, tools
from deap.benchmarks import tools as btools

warnings.filterwarnings("ignore")

# ================================================================
# FIXED — Individual types and generic utilities (do not modify)
# ================================================================

# Create DEAP fitness and individual types
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

# For 3-objective problems
creator.create("FitnessMin3", base.Fitness, weights=(-1.0, -1.0, -1.0))
creator.create("Individual3", list, fitness=creator.FitnessMin3)


def make_individual(n_var, bounds, ind_class):
    """Create a random individual within bounds."""
    lo, hi = bounds
    return ind_class([random.uniform(lo, hi) for _ in range(n_var)])


def evaluate(individual, func):
    """Evaluate an individual on the (held-out) objective function."""
    return func(individual)


def bounded_crossover(ind1, ind2, eta, low, up):
    """Simulated Binary Crossover (SBX) with bounds."""
    tools.cxSimulatedBinaryBounded(ind1, ind2, eta=eta, low=low, up=up)
    return ind1, ind2


def bounded_mutation(individual, eta, low, up, indpb):
    """Polynomial mutation with bounds."""
    tools.mutPolynomialBounded(individual, eta=eta, low=low, up=up, indpb=indpb)
    return (individual,)


def get_nondominated(population):
    """Extract the first non-dominated front from the population."""
    pareto_fronts = tools.sortNondominated(population, len(population), first_front_only=True)
    return pareto_fronts[0]


def compute_crowding_distance(individuals):
    """Compute crowding distance for a set of individuals."""
    if len(individuals) <= 2:
        for ind in individuals:
            ind.fitness.crowding_dist = float("inf")
        return
    n_obj = len(individuals[0].fitness.values)
    for ind in individuals:
        ind.fitness.crowding_dist = 0.0
    for m in range(n_obj):
        individuals.sort(key=lambda x: x.fitness.values[m])
        individuals[0].fitness.crowding_dist = float("inf")
        individuals[-1].fitness.crowding_dist = float("inf")
        f_max = individuals[-1].fitness.values[m]
        f_min = individuals[0].fitness.values[m]
        if f_max - f_min < 1e-12:
            continue
        for i in range(1, len(individuals) - 1):
            individuals[i].fitness.crowding_dist += (
                individuals[i + 1].fitness.values[m] - individuals[i - 1].fitness.values[m]
            ) / (f_max - f_min)


# ================================================================
# FIXED — Held-out problem spec loading (do not modify)
# ================================================================
#
# The harness pre-generates, for the opaque problem key in ENV, a spec file
# carrying the numeric problem configuration and an opaque black-box objective
# evaluator f(individual) -> objectives. This program loads that spec, builds the
# evaluator as a pure black box, and uses it to evaluate candidate solutions.
# The problem name, the analytic Pareto front, and the metrics are NOT present
# here — they live in a host-only module the agent's process cannot import. The
# host-side scorer regenerates the front and computes HV/IGD/Spread.


def _spec_dir():
    d = os.environ.get("MOEA_SPEC_DIR")
    if d:
        return d
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_moea_specs")


def _load_spec(env_key, seed):
    path = os.path.join(_spec_dir(), f"{env_key}_seed{seed}.json.b64")
    with open(path, "r") as f:
        raw = base64.b64decode(f.read())
    return json.loads(raw.decode("utf-8"))


def _build_objective(spec):
    """Reconstruct the black-box objective f(individual) -> tuple from the spec.

    The evaluator is a marshalled, name-free, problem-specific code object that
    inlines the objective arithmetic; it is used purely as a black box and
    carries no problem identity (no name, no ``kind``) the strategy could exploit.
    """
    code = marshal.loads(base64.b64decode(spec["evaluator"]))
    kernel = types.FunctionType(code, {"__builtins__": __builtins__}, "objective")
    # No problem id in the spec; the kernel is specific to this run's problem.
    n_obj = int(spec["n_obj"])

    def f(individual):
        return tuple(kernel(individual, n_obj))

    return f


# ================================================================
# EDITABLE — Custom multi-objective evolutionary strategy (lines 159 to 303)
# The agent modifies ONLY this section.
# ================================================================


class CustomMOEA:
    """Custom multi-objective evolutionary algorithm.

    The agent should implement a novel evolutionary strategy for multi-objective
    optimization. The algorithm operates on a population of individuals, each
    with a fitness consisting of multiple objective values (all minimized).

    Available DEAP utilities (already imported):
        - tools.sortNondominated(pop, k) -> list of fronts
        - tools.selTournamentDCD(pop, k) -> selected individuals
        - tools.cxSimulatedBinaryBounded(ind1, ind2, eta, low, up)
        - tools.mutPolynomialBounded(ind, eta, low, up, indpb)
        - tools.uniform_reference_points(nobj, p) -> reference points array
        - compute_crowding_distance(individuals) -> sets .fitness.crowding_dist
        - get_nondominated(population) -> first front

    Individual interface:
        ind.fitness.values -> tuple of objective values (all minimized)
        ind.fitness.dominates(other.fitness) -> bool
        ind.fitness.valid -> bool (True if evaluated)

    Args:
        pop_size: population size
        n_obj: number of objectives
        n_var: number of decision variables
        bounds: (low, high) for all variables
        cx_eta: SBX crossover distribution index (default 20)
        mut_eta: polynomial mutation distribution index (default 20)
        mut_prob: per-variable mutation probability (default 1/n_var)
    """

    def __init__(
        self,
        pop_size: int,
        n_obj: int,
        n_var: int,
        bounds: Tuple[float, float],
        cx_eta: float = 20.0,
        mut_eta: float = 20.0,
        mut_prob: Optional[float] = None,
    ):
        self.pop_size = pop_size
        self.n_obj = n_obj
        self.n_var = n_var
        self.bounds = bounds
        self.cx_eta = cx_eta
        self.mut_eta = mut_eta
        self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var

    def select(self, population: list, k: int) -> list:
        """Select k parents from the population for mating.

        Default: binary tournament selection based on non-domination rank
        and crowding distance (NSGA-II style). Replace with a better strategy.

        Args:
            population: current population (list of Individuals)
            k: number of parents to select
        Returns:
            list of k selected individuals (copies)
        """
        # Assign crowding distances for tournament selection
        fronts = tools.sortNondominated(population, len(population), first_front_only=False)
        for front in fronts:
            compute_crowding_distance(front)
        return tools.selTournamentDCD(population, k)

    def vary(self, parents: list) -> list:
        """Apply crossover and mutation to produce offspring.

        Default: SBX crossover (probability 0.9) + polynomial mutation.
        Replace or augment with novel variation operators.

        Args:
            parents: list of selected parent individuals
        Returns:
            list of offspring individuals (fitness invalidated)
        """
        offspring = [deepcopy(ind) for ind in parents]
        lo, hi = self.bounds

        # Pairwise crossover
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < 0.9:
                tools.cxSimulatedBinaryBounded(
                    offspring[i], offspring[i + 1],
                    eta=self.cx_eta, low=lo, up=hi,
                )
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values

        # Mutation
        for ind in offspring:
            if random.random() < 1.0:
                tools.mutPolynomialBounded(
                    ind, eta=self.mut_eta, low=lo, up=hi, indpb=self.mut_prob,
                )
                del ind.fitness.values

        return offspring

    def survive(self, population: list, offspring: list) -> list:
        """Environmental selection: choose next generation from combined pool.

        Default: NSGA-II survival — non-dominated sorting + crowding distance.
        Replace with a better environmental selection mechanism.

        Args:
            population: current population
            offspring: newly generated offspring
        Returns:
            list of pop_size individuals for the next generation
        """
        combined = population + offspring

        # Non-dominated sorting
        fronts = tools.sortNondominated(combined, self.pop_size, first_front_only=False)

        next_gen = []
        for front in fronts:
            if len(next_gen) + len(front) <= self.pop_size:
                next_gen.extend(front)
            else:
                # Fill remaining slots using crowding distance
                remaining = self.pop_size - len(next_gen)
                compute_crowding_distance(front)
                front.sort(key=lambda x: x.fitness.crowding_dist, reverse=True)
                next_gen.extend(front[:remaining])
                break

        return next_gen

    def on_generation(self, gen: int, population: list):
        """Optional callback at the end of each generation.

        Can be used for adaptive parameter updates, archive maintenance, etc.
        Default: no-op.

        Args:
            gen: current generation number (1-indexed)
            population: current population after survival selection
        """
        pass


# ================================================================
# FIXED — Main evolution loop and prediction emit (do not modify below)
# ================================================================


def _check_spec_magic(spec):
    """Refuse a marshalled evaluator written by an incompatible CPython.

    marshal does not validate the producer's CPython version: a blob written by
    a different interpreter loads WITHOUT error and then crashes the process
    (SIGSEGV) on first call. The spec records the producer interpreter's magic
    number; a mismatch means the spec was generated by a different CPython than
    the one running this program, so fail here with a clear, diagnosable error
    instead of loading foreign bytecode in ``_build_objective``.
    """
    import importlib.util

    producer_magic = spec.get("producer_magic")
    runtime_magic = importlib.util.MAGIC_NUMBER.hex()
    if producer_magic != runtime_magic:
        raise RuntimeError(
            "the marshalled objective in this problem spec was produced by an "
            f"incompatible CPython (spec magic={producer_magic!r}, this "
            f"interpreter magic={runtime_magic!r}); the specs must be "
            "(re)generated with the same CPython version that runs this program"
        )


def run_moea(env_key: str, seed: int, output_dir: str):
    """Run the custom MOEA on the held-out benchmark problem.

    Loads the pre-generated problem spec for ``env_key``, runs the strategy, and
    emits the final non-dominated population's objective values for the host-side
    scorer. The true Pareto front and the metrics are computed host-side; this
    process never sees them.
    """
    spec = _load_spec(env_key, seed)
    _check_spec_magic(spec)
    n_var = int(spec["n_var"])
    n_obj = int(spec["n_obj"])
    bounds = tuple(spec["bounds"])
    pop_size = int(spec["pop_size"])
    n_gen = int(spec["n_gen"])

    # Black-box objective evaluator (legitimate: evaluating candidates is the task)
    func = _build_objective(spec)

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)

    # Determine individual class based on number of objectives
    ind_class = creator.Individual3 if n_obj == 3 else creator.Individual

    # Initialize algorithm
    moea = CustomMOEA(
        pop_size=pop_size,
        n_obj=n_obj,
        n_var=n_var,
        bounds=bounds,
    )

    # Create initial population
    population = [make_individual(n_var, bounds, ind_class) for _ in range(pop_size)]

    # Evaluate initial population
    for ind in population:
        ind.fitness.values = evaluate(ind, func)

    for gen in range(1, n_gen + 1):
        # Parent selection
        parents = moea.select(population, pop_size)

        # Variation (crossover + mutation)
        offspring = moea.vary(parents)

        # Evaluate offspring
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = evaluate(ind, func)

        # Environmental selection (survival)
        population = moea.survive(population, offspring)

        # Optional per-generation callback
        moea.on_generation(gen, population)

        # Periodic progress feedback (objective-space extent only, no metrics)
        if gen % 20 == 0 or gen == n_gen:
            nd_front = get_nondominated(population)
            front_values = np.array([ind.fitness.values for ind in nd_front])
            print(
                f"TRAIN_PROGRESS gen={gen} front_size={len(nd_front)} "
                f"f_min={np.min(front_values, axis=0).round(4).tolist()} "
                f"f_max={np.max(front_values, axis=0).round(4).tolist()}",
                flush=True,
            )

    # Final non-dominated front
    nd_front = get_nondominated(population)
    front_values = np.array([ind.fitness.values for ind in nd_front], dtype=np.float64)

    # Emit the final population's objective values for the host-side scorer. We do
    # NOT have the true Pareto front, so we cannot (and do not) compute metrics.
    payload = base64.b64encode(
        np.ascontiguousarray(front_values, dtype=np.float64).tobytes()
    ).decode("ascii")
    print(
        f"MOEA_PRED env={env_key} seed={seed} shape={front_values.shape[0]},{front_values.shape[1]} "
        f"objs={payload}",
        flush=True,
    )

    # Save final front to disk (objective values only)
    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(
        os.path.join(output_dir, f"{env_key}_front.csv"),
        front_values,
        delimiter=",",
        header=",".join(f"f{i+1}" for i in range(n_obj)),
    )

    return front_values


def main():
    parser = argparse.ArgumentParser(description="Multi-Objective Optimization Benchmark")
    parser.add_argument("--env", type=str, default=os.environ.get("ENV", ""))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", 42)))
    parser.add_argument("--output-dir", type=str, default=os.environ.get("OUTPUT_DIR", "./output"))
    args = parser.parse_args()

    if not args.env:
        raise SystemExit("ENV not set")

    print(f"Running MOEA benchmark: {args.env} (seed={args.seed})", flush=True)
    run_moea(args.env, args.seed, args.output_dir)
    print(f"Done {args.env}.", flush=True)


if __name__ == "__main__":
    main()
