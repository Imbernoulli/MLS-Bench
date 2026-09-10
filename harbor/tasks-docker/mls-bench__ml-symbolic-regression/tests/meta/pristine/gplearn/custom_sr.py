#!/usr/bin/env python3
"""Symbolic Regression via Genetic Programming.

A self-contained GP framework for symbolic regression. The editable section
contains the search strategy: fitness function, selection, crossover, mutation,
and per-generation evolution logic.

The benchmark identity (which target function is used) and the held-out test
labels are NOT available here. The FIXED runner loads a pre-generated
``(X_train, y_train, X_test)`` triple — only SAMPLES of the target on the
training inputs — drives the GP search using those samples for fitness, then
emits the best evolved expression's predictions on ``X_test``. The host-side
scorer regenerates the test labels and computes R2. Your GP must fit the
``(X_train, y_train)`` samples; there is no closed-form target to read off.
"""

import argparse
import base64
import io
import math
import os
import random
import sys

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


def _train_r2(y_true, y_pred):
    """R2 of the GP fit on the TRAINING samples (feedback only).

    This uses only the (X_train, y_train) samples the search already has access
    to, so it leaks nothing about the held-out test target. The official test
    R2 is computed host-side from the emitted predictions.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-15:
        return 1.0 if ss_res < 1e-15 else 0.0
    return max(1.0 - ss_res / ss_tot, 0.0)


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
# FIXED: input loading + GP driver + prediction emit
# (do not modify below this line)
# ============================================================

def _inputs_dir():
    d = os.environ.get("SR_INPUTS_DIR")
    if d:
        return d
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_sr_inputs")


def _load_input(task_id, seed):
    """Load the pre-generated (X_train, y_train, X_test) for this run.

    Only training SAMPLES (X_train, y_train) and the test inputs (X_test) are
    present; the closed-form target and the test labels are withheld.
    """
    path = os.path.join(_inputs_dir(), f"{task_id}_seed{seed}.npz.b64")
    with open(path, "r") as f:
        raw = base64.b64decode(f.read())
    d = np.load(io.BytesIO(raw))
    return d["X_train"], d["y_train"], d["X_test"]


def main():
    parser = argparse.ArgumentParser(description="GP Symbolic Regression")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pop-size', type=int, default=500)
    parser.add_argument('--generations', type=int, default=50)
    parser.add_argument('--max-depth', type=int, default=6)
    args = parser.parse_args()

    # Opaque task id used only to locate the pre-generated inputs; it carries
    # no information about which target function is in use.
    task_id = os.environ.get("SR_TASK", "")
    if not task_id:
        raise SystemExit("SR_TASK not set")

    random.seed(args.seed)
    np.random.seed(args.seed)

    X_train, y_train, X_test = _load_input(task_id, args.seed)
    n_features = X_train.shape[1]

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
        train_r2 = _train_r2(y_train, y_pred_gen)

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

    # Final fit summary on the training samples (feedback only)
    y_pred_train = safe_evaluate(best_tree_ever, X_train)
    train_r2 = _train_r2(y_train, y_pred_train)
    expr_str = str(best_tree_ever)

    # Emit predictions on the held-out test inputs for host-side scoring.
    y_pred_test = safe_evaluate(best_tree_ever, X_test)
    y_pred_test = np.ascontiguousarray(np.asarray(y_pred_test, dtype=np.float64)).ravel()
    payload = base64.b64encode(y_pred_test.tobytes()).decode("ascii")

    print(
        f"TEST_METRICS train_r2={train_r2:.6f} size={best_tree_ever.size()} "
        f'expression="{expr_str}"',
        flush=True,
    )
    print(
        f"SR_PRED task={task_id} seed={args.seed} n={y_pred_test.shape[0]} "
        f"preds={payload}",
        flush=True,
    )


if __name__ == '__main__':
    main()
