"""
Multi-Objective Optimization — Custom Evolutionary Strategy Template

This script runs a complete multi-objective evolutionary algorithm on standard
benchmark problems (ZDT/DTLZ). The agent should implement the custom selection
and variation strategy in the CustomMOEA class.

Usage:
    python deap/custom_moea.py --problem zdt1 --seed 42 --output-dir ./out
"""

import argparse
import json
import math
import os
import random
import time
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
# FIXED — Problem definitions and utilities (do not modify)
# ================================================================

# Create DEAP fitness and individual types
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

# For 3-objective problems
creator.create("FitnessMin3", base.Fitness, weights=(-1.0, -1.0, -1.0))
creator.create("Individual3", list, fitness=creator.FitnessMin3)


PROBLEMS = {
    "zdt1": {
        "func": benchmarks.zdt1,
        "n_var": 30,
        "n_obj": 2,
        "bounds": (0.0, 1.0),
        "pop_size": 100,
        "n_gen": 200,
        "ref_point": [1.1, 1.1],
        "description": "ZDT1: convex Pareto front, 30 variables, 2 objectives",
    },
    "zdt3": {
        "func": benchmarks.zdt3,
        "n_var": 30,
        "n_obj": 2,
        "bounds": (0.0, 1.0),
        "pop_size": 100,
        "n_gen": 200,
        "ref_point": [1.1, 1.1],
        "description": "ZDT3: disconnected Pareto front, 30 variables, 2 objectives",
    },
    "dtlz2": {
        "func": lambda ind: benchmarks.dtlz2(ind, 3),
        "n_var": 12,
        "n_obj": 3,
        "bounds": (0.0, 1.0),
        "pop_size": 120,
        "n_gen": 250,
        "ref_point": [1.5, 1.5, 1.5],
        "description": "DTLZ2: spherical Pareto front, 12 variables, 3 objectives",
    },
    "dtlz1": {
        "func": lambda ind: benchmarks.dtlz1(ind, 3),
        "n_var": 7,
        "n_obj": 3,
        "bounds": (0.0, 1.0),
        "pop_size": 120,
        "n_gen": 400,
        "ref_point": [1.0, 1.0, 1.0],
        "description": "DTLZ1: linear Pareto front with many local fronts, 7 variables, 3 objectives",
    },
}


def generate_pareto_front(problem_name: str, n_points: int = 500) -> np.ndarray:
    """Generate reference Pareto front points for IGD computation."""
    if problem_name == "zdt1":
        x = np.linspace(0, 1, n_points)
        return np.column_stack([x, 1 - np.sqrt(x)])
    elif problem_name == "zdt3":
        # ZDT3 has a disconnected front
        regions = [
            (0.0, 0.0830),
            (0.1822, 0.2577),
            (0.4093, 0.4538),
            (0.6183, 0.6525),
            (0.8233, 0.8518),
        ]
        points = []
        per_region = n_points // len(regions)
        for lo, hi in regions:
            x = np.linspace(lo, hi, per_region)
            f1 = x
            f2 = 1 - np.sqrt(x) - x * np.sin(10 * np.pi * x)
            points.append(np.column_stack([f1, f2]))
        return np.vstack(points)
    elif problem_name == "dtlz2":
        # Uniform points on first octant of unit sphere
        points = []
        ns = int(np.sqrt(n_points)) + 1
        for i in range(ns):
            for j in range(ns):
                theta1 = (i / max(ns - 1, 1)) * np.pi / 2
                theta2 = (j / max(ns - 1, 1)) * np.pi / 2
                f1 = np.cos(theta1) * np.cos(theta2)
                f2 = np.cos(theta1) * np.sin(theta2)
                f3 = np.sin(theta1)
                points.append([f1, f2, f3])
        return np.array(points[:n_points])
    elif problem_name == "dtlz1":
        # Pareto front lies on the plane sum(f_i) = 0.5
        points = []
        ns = int(np.sqrt(n_points)) + 1
        for i in range(ns):
            for j in range(ns - i):
                f1 = i / max(ns - 1, 1) * 0.5
                f2 = j / max(ns - 1, 1) * 0.5
                f3 = 0.5 - f1 - f2
                if f3 >= -1e-8:
                    points.append([f1, f2, max(f3, 0.0)])
        return np.array(points[:n_points])
    else:
        raise ValueError(f"Unknown problem: {problem_name}")


def _hv_2d(points, ref):
    """Exact 2D hypervolume via non-dominated sweep."""
    pts = points[(points[:, 0] < ref[0]) & (points[:, 1] < ref[1])]
    if len(pts) == 0:
        return 0.0
    pts = pts[pts[:, 0].argsort()]
    nd = [pts[0]]
    for p in pts[1:]:
        if p[1] < nd[-1][1]:
            nd.append(p)
    nd = np.array(nd)
    hv = 0.0
    prev_y = ref[1]
    for p in nd:
        width = ref[0] - p[0]
        hv += width * (prev_y - p[1])
        prev_y = p[1]
    return hv


def _hv_3d(points, ref):
    """Exact 3D hypervolume via z-slicing + 2D sweep."""
    mask = np.all(points < ref, axis=1)
    pts = points[mask]
    if len(pts) == 0:
        return 0.0
    # Sort by z ascending: as z increases, more points become active
    order = np.argsort(pts[:, 2])
    pts = pts[order]
    hv = 0.0
    active_2d = []
    for i in range(len(pts)):
        active_2d.append(pts[i, :2])
        z_lo = pts[i, 2]
        z_hi = pts[i + 1, 2] if i + 1 < len(pts) else ref[2]
        dz = z_hi - z_lo
        if dz > 0:
            hv += _hv_2d(np.array(active_2d), ref[:2]) * dz
    return hv


def compute_hypervolume(nd_front, ref_point):
    """Robust hypervolume computation that works for 2D and 3D.

    Falls back to a pure-Python implementation if DEAP's built-in fails.
    """
    # Always use pure-Python implementation (DEAP's C version fails silently in some envs)
    front_values = np.array([ind.fitness.values for ind in nd_front])
    ref = np.array(ref_point, dtype=np.float64)
    # Filter out points not dominated by ref
    mask = np.all(front_values < ref, axis=1)
    front_values = front_values[mask]
    if len(front_values) == 0:
        return 0.0
    n_obj = front_values.shape[1]
    if n_obj == 2:
        return _hv_2d(front_values, ref)
    elif n_obj == 3:
        return _hv_3d(front_values, ref)
    return 0.0


def compute_spread(front_values: np.ndarray) -> float:
    """Compute spread (Delta) metric for a 2D front.

    Measures the extent and uniformity of the Pareto front approximation.
    Lower is better. For >2 objectives, returns average pairwise distance std.
    """
    if len(front_values) < 2:
        return float("inf")

    n_obj = front_values.shape[1]
    if n_obj == 2:
        # Sort by first objective
        sorted_idx = np.argsort(front_values[:, 0])
        sorted_front = front_values[sorted_idx]
        # Consecutive distances
        dists = np.sqrt(np.sum(np.diff(sorted_front, axis=0) ** 2, axis=1))
        if len(dists) == 0:
            return float("inf")
        d_mean = np.mean(dists)
        if d_mean < 1e-12:
            return float("inf")
        spread = np.sum(np.abs(dists - d_mean)) / (len(dists) * d_mean)
        return float(spread)
    else:
        # For many-objective: use spacing metric
        from scipy.spatial.distance import cdist

        dist_matrix = cdist(front_values, front_values)
        np.fill_diagonal(dist_matrix, np.inf)
        min_dists = np.min(dist_matrix, axis=1)
        d_mean = np.mean(min_dists)
        if d_mean < 1e-12:
            return float("inf")
        spread = np.sqrt(np.mean((min_dists - d_mean) ** 2)) / d_mean
        return float(spread)


def make_individual(n_var, bounds, ind_class):
    """Create a random individual within bounds."""
    lo, hi = bounds
    return ind_class([random.uniform(lo, hi) for _ in range(n_var)])


def evaluate(individual, func):
    """Evaluate an individual on the benchmark function."""
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
# EDITABLE — Custom multi-objective evolutionary strategy (lines 297 to 446)
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
# FIXED — Main evolution loop and evaluation (do not modify below)
# ================================================================


def run_moea(problem_name: str, seed: int, output_dir: str):
    """Run the custom MOEA on a benchmark problem."""
    cfg = PROBLEMS[problem_name]
    func = cfg["func"]
    n_var = cfg["n_var"]
    n_obj = cfg["n_obj"]
    bounds = cfg["bounds"]
    pop_size = cfg["pop_size"]
    n_gen = cfg["n_gen"]
    ref_point = cfg["ref_point"]

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

    # Generate reference Pareto front for IGD
    pf_ref = generate_pareto_front(problem_name, n_points=500)

    # Track metrics over generations
    hv_history = []
    igd_history = []

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

        # Compute metrics periodically
        if gen % 20 == 0 or gen == n_gen:
            nd_front = get_nondominated(population)
            front_values = np.array([ind.fitness.values for ind in nd_front])

            # Hypervolume (robust computation with fallback)
            hv_val = compute_hypervolume(nd_front, ref_point)

            # IGD
            try:
                igd_val = btools.igd(front_values, pf_ref)
            except Exception:
                igd_val = float("inf")

            # Spread
            spread_val = compute_spread(front_values)

            hv_history.append(hv_val)
            igd_history.append(igd_val)

            print(
                f"TRAIN_METRICS gen={gen} hv={hv_val:.6f} igd={igd_val:.6f} "
                f"spread={spread_val:.6f} front_size={len(nd_front)}",
                flush=True,
            )

    # Final evaluation
    nd_front = get_nondominated(population)
    front_values = np.array([ind.fitness.values for ind in nd_front])

    final_hv = compute_hypervolume(nd_front, ref_point)

    try:
        final_igd = btools.igd(front_values, pf_ref)
    except Exception:
        final_igd = float("inf")

    final_spread = compute_spread(front_values)

    print(f"TEST_METRICS hv={final_hv:.6f} igd={final_igd:.6f} spread={final_spread:.6f}", flush=True)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results = {
        "problem": problem_name,
        "seed": seed,
        "n_var": n_var,
        "n_obj": n_obj,
        "pop_size": pop_size,
        "n_gen": n_gen,
        "final_hv": final_hv,
        "final_igd": final_igd,
        "final_spread": final_spread,
        "front_size": len(nd_front),
        "hv_history": hv_history,
        "igd_history": igd_history,
    }
    with open(os.path.join(output_dir, f"{problem_name}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save final front
    np.savetxt(
        os.path.join(output_dir, f"{problem_name}_front.csv"),
        front_values,
        delimiter=",",
        header=",".join(f"f{i+1}" for i in range(n_obj)),
    )

    return final_hv, final_igd, final_spread


def main():
    parser = argparse.ArgumentParser(description="Multi-Objective Optimization Benchmark")
    parser.add_argument("--problem", type=str, required=True, choices=list(PROBLEMS.keys()))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", 42)))
    parser.add_argument("--output-dir", type=str, default=os.environ.get("OUTPUT_DIR", "./output"))
    args = parser.parse_args()

    print(f"Running MOEA benchmark: {args.problem} (seed={args.seed})", flush=True)
    print(f"  {PROBLEMS[args.problem]['description']}", flush=True)
    hv, igd, spread = run_moea(args.problem, args.seed, args.output_dir)
    print(f"Final on {args.problem}: HV={hv:.6f} IGD={igd:.6f} Spread={spread:.6f}", flush=True)


if __name__ == "__main__":
    main()
