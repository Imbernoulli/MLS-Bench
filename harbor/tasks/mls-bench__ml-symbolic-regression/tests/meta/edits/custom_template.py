#!/usr/bin/env python3
"""Symbolic Regression via Genetic Programming.

A self-contained GP framework for symbolic regression benchmarks.
The editable section contains the search strategy: fitness function,
selection, crossover, mutation, and per-generation evolution logic.
"""

import argparse
import math
import random
import sys
import os
import numpy as np


# ============================================================
# Operator Definitions (FIXED)
# ============================================================

def protected_div(a, b):
    """Protected division: returns 1.0 when divisor is near zero."""
    return np.where(np.abs(b) > 1e-10, a / b, 1.0)


def protected_log(a):
    """Protected log: returns 0.0 for non-positive inputs."""
    return np.where(np.abs(a) > 1e-10, np.log(np.abs(a)), 0.0)


def protected_exp(a):
    """Protected exp: clips input to prevent overflow."""
    return np.exp(np.clip(a, -10, 10))


OPERATORS = {
    'add': (np.add, 2),
    'sub': (np.subtract, 2),
    'mul': (np.multiply, 2),
    'div': (protected_div, 2),
    'sin': (np.sin, 1),
    'cos': (np.cos, 1),
    'log': (protected_log, 1),
    'exp': (protected_exp, 1),
}

OPERATOR_NAMES = list(OPERATORS.keys())


# ============================================================
# Tree Representation (FIXED)
# ============================================================

class Node:
    """A node in the GP expression tree."""
    __slots__ = ('value', 'children')

    def __init__(self, value, children=None):
        self.value = value
        self.children = children or []

    @property
    def is_terminal(self):
        return len(self.children) == 0

    def evaluate(self, X):
        """Evaluate expression tree on input array X (n_samples, n_features)."""
        if self.is_terminal:
            if isinstance(self.value, str) and self.value.startswith('x'):
                idx = int(self.value[1:])
                return X[:, idx].copy()
            else:
                return np.full(X.shape[0], float(self.value))
        func, arity = OPERATORS[self.value]
        args = [child.evaluate(X) for child in self.children]
        result = func(*args)
        return np.clip(result, -1e15, 1e15)

    def size(self):
        """Count total nodes in the tree."""
        return 1 + sum(c.size() for c in self.children)

    def depth(self):
        """Compute tree depth."""
        if not self.children:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def copy(self):
        """Deep copy the tree."""
        return Node(self.value, [c.copy() for c in self.children])

    def get_all_nodes(self):
        """Return a list of (node, parent, child_index) via preorder traversal."""
        result = [(self, None, None)]
        for i, child in enumerate(self.children):
            child_nodes = child.get_all_nodes()
            # Update parent info for direct children
            child_nodes[0] = (child, self, i)
            result.extend(child_nodes)
        return result

    def __str__(self):
        if self.is_terminal:
            return str(self.value)
        if len(self.children) == 1:
            return f"{self.value}({self.children[0]})"
        return f"({self.children[0]} {self.value} {self.children[1]})"


# ============================================================
# Tree Generation (FIXED)
# ============================================================

def random_terminal(n_features, const_range=(-5.0, 5.0)):
    """Generate a random terminal node (variable or constant)."""
    if random.random() < 0.5:
        idx = random.randint(0, n_features - 1)
        return Node(f'x{idx}')
    else:
        return Node(str(round(random.uniform(*const_range), 2)))


def generate_tree(method, max_depth, n_features, depth=0):
    """Generate a random expression tree using 'grow' or 'full' method."""
    if depth >= max_depth or (method == 'grow' and depth > 0 and random.random() < 0.3):
        return random_terminal(n_features)
    op_name = random.choice(OPERATOR_NAMES)
    _, arity = OPERATORS[op_name]
    children = [generate_tree(method, max_depth, n_features, depth + 1)
                for _ in range(arity)]
    return Node(op_name, children)


def ramped_half_and_half(pop_size, max_depth, n_features):
    """Initialize population with ramped half-and-half method."""
    population = []
    for i in range(pop_size):
        depth = 2 + (i % (max_depth - 1))
        method = 'full' if i % 2 == 0 else 'grow'
        population.append(generate_tree(method, depth, n_features))
    return population


# ============================================================
# Benchmark Data (FIXED)
# ============================================================

BENCHMARKS = {
    'nguyen7': {
        'func': lambda X: np.log(X[:, 0] + 1) + np.log(X[:, 0] ** 2 + 1),
        'n_features': 1,
        'train_range': (0.0, 2.0),
        'n_train': 20,
        'test_range': (-0.5, 2.5),
        'n_test': 100,
    },
    'nguyen10': {
        'func': lambda X: 2 * np.sin(X[:, 0]) * np.cos(X[:, 1]),
        'n_features': 2,
        'train_range': (0.0, 2 * np.pi),
        'n_train': 100,
        'test_range': (0.0, 2 * np.pi),
        'n_test': 400,
    },
    'koza3': {
        'func': lambda X: X[:, 0] ** 5 - 2 * X[:, 0] ** 3 + X[:, 0],
        'n_features': 1,
        'train_range': (-1.0, 1.0),
        'n_train': 20,
        'test_range': (-1.0, 1.0),
        'n_test': 100,
    },
}


def generate_data(benchmark_name, seed=42):
    """Generate train/test data for a benchmark function."""
    bench = BENCHMARKS[benchmark_name]
    n_features = bench['n_features']
    lo, hi = bench['train_range']

    if n_features == 1:
        X_train = np.linspace(lo, hi, bench['n_train']).reshape(-1, 1)
        lo_t, hi_t = bench['test_range']
        X_test = np.linspace(lo_t, hi_t, bench['n_test']).reshape(-1, 1)
    else:
        n_per_dim = int(round(bench['n_train'] ** (1.0 / n_features)))
        grids = [np.linspace(lo, hi, n_per_dim) for _ in range(n_features)]
        X_train = np.array(np.meshgrid(*grids)).T.reshape(-1, n_features)
        lo_t, hi_t = bench['test_range']
        n_per_dim_t = int(round(bench['n_test'] ** (1.0 / n_features)))
        grids_t = [np.linspace(lo_t, hi_t, n_per_dim_t) for _ in range(n_features)]
        X_test = np.array(np.meshgrid(*grids_t)).T.reshape(-1, n_features)

    y_train = bench['func'](X_train)
    y_test = bench['func'](X_test)
    return X_train, y_train, X_test, y_test, n_features


# ============================================================
# Evaluation Utilities (FIXED)
# ============================================================

def safe_evaluate(tree, X):
    """Evaluate tree with error handling."""
    try:
        result = tree.evaluate(X)
        result = np.nan_to_num(result, nan=1e10, posinf=1e10, neginf=-1e10)
        return np.clip(result, -1e10, 1e10)
    except Exception:
        return np.full(X.shape[0], 1e10)


def r2_score(y_true, y_pred):
    """Compute R-squared score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-15:
        return 1.0 if ss_res < 1e-15 else 0.0
    return max(1.0 - ss_res / ss_tot, 0.0)  # SRBench floor: clip blowups


# ============================================================
# Search Strategy (EDITABLE)
# ============================================================

def fitness_function(tree, X, y):
    """Evaluate fitness of a candidate program. Lower is better."""
    y_pred = safe_evaluate(tree, X)
    return float(np.mean((y - y_pred) ** 2))


def selection(population, fitnesses, n_select):
    """Select individuals from the population for reproduction.

    Args:
        population: list of Node trees
        fitnesses: list of float fitness values (lower is better)
        n_select: int number of individuals to select

    Returns:
        list of Node copies of selected individuals
    """
    selected = []
    for _ in range(n_select):
        idx = random.randint(0, len(population) - 1)
        selected.append(population[idx].copy())
    return selected


def crossover(parent1, parent2, n_features, max_depth=17):
    """Perform crossover between two parent trees.

    Returns:
        Node - offspring tree
    """
    return parent1.copy()


def mutation(parent, n_features, max_depth=17):
    """Perform mutation on a parent tree.

    Returns:
        Node - mutated tree
    """
    return parent.copy()


def evolve_one_generation(population, fitnesses, X_train, y_train,
                          n_features, pop_size,
                          crossover_rate=0.9, mutation_rate=0.05,
                          max_depth=17):
    """Create the next generation from the current population.

    Args:
        population: list of Node trees
        fitnesses: list of float fitness values (lower is better)
        X_train: numpy array (n_samples, n_features) - training inputs
        y_train: numpy array (n_samples,) - training targets
        n_features: number of input features
        pop_size: desired population size
        crossover_rate: probability of crossover
        mutation_rate: probability of mutation
        max_depth: maximum allowed tree depth

    Returns:
        list of Node - next generation population
    """
    new_population = []
    # Elitism: keep best individual
    elite_idx = int(np.argmin(fitnesses))
    new_population.append(population[elite_idx].copy())

    while len(new_population) < pop_size:
        parents = selection(population, fitnesses, 2)
        r = random.random()
        if r < crossover_rate:
            child = crossover(parents[0], parents[1], n_features, max_depth)
        elif r < crossover_rate + mutation_rate:
            child = mutation(parents[0], n_features, max_depth)
        else:
            child = parents[0]
        new_population.append(child)

    return new_population[:pop_size]


# ============================================================
# Main GP Loop (FIXED)
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="GP Symbolic Regression")
    parser.add_argument('--benchmark', type=str, required=True,
                        choices=list(BENCHMARKS.keys()))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pop-size', type=int, default=500)
    parser.add_argument('--generations', type=int, default=50)
    parser.add_argument('--max-depth', type=int, default=6)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    X_train, y_train, X_test, y_test, n_features = generate_data(
        args.benchmark, args.seed
    )

    # Initialize population
    population = ramped_half_and_half(args.pop_size, args.max_depth, n_features)

    best_fitness_ever = float('inf')
    best_tree_ever = None

    for gen in range(args.generations):
        fitnesses = [fitness_function(tree, X_train, y_train)
                     for tree in population]

        best_idx = int(np.argmin(fitnesses))
        best_fitness = fitnesses[best_idx]
        avg_fitness = float(np.mean(fitnesses))
        best_size = population[best_idx].size()

        if best_fitness < best_fitness_ever:
            best_fitness_ever = best_fitness
            best_tree_ever = population[best_idx].copy()

        y_pred_gen = safe_evaluate(best_tree_ever, X_train)
        train_r2 = r2_score(y_train, y_pred_gen)

        print(
            f"TRAIN_METRICS generation={gen} best_fitness={best_fitness:.6f} "
            f"avg_fitness={avg_fitness:.6f} best_size={best_size} "
            f"train_r2={train_r2:.6f}",
            flush=True,
        )

        if gen < args.generations - 1:
            population = evolve_one_generation(
                population, fitnesses, X_train, y_train,
                n_features, args.pop_size,
                max_depth=args.max_depth + 2,
            )

    # Final evaluation on test set
    y_pred_train = safe_evaluate(best_tree_ever, X_train)
    y_pred_test = safe_evaluate(best_tree_ever, X_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = float(np.sqrt(np.mean((y_test - y_pred_test) ** 2)))
    expr_str = str(best_tree_ever)

    print(
        f"TEST_METRICS r2={test_r2:.6f} rmse={test_rmse:.6f} "
        f"train_r2={train_r2:.6f} size={best_tree_ever.size()} "
        f'expression="{expr_str}"',
        flush=True,
    )


if __name__ == '__main__':
    main()
