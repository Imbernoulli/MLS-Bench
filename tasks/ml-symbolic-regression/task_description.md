# Symbolic Regression: GP Search Strategy

## Research Question
Design a genetic-programming search strategy for symbolic regression that more reliably discovers symbolic expressions fitting the target function. The contribution is the *search strategy itself*: fitness shaping, parent selection, crossover/mutation operators, elitism, parsimony pressure, diversity maintenance, or adaptive operator rates.

## Background
Symbolic regression searches the space of mathematical expressions for one that fits observed `(X, y)` data. Genetic programming (GP) maintains a population of expression trees and evolves them by selection, crossover, and mutation. The central tensions are exploration vs. exploitation, controlling expression complexity (bloat), and avoiding premature convergence to local optima.

Reference baselines (provided as `edit_ops` over the same `custom_sr.py` skeleton):
- **Standard GP** — Koza, *Genetic Programming*, MIT Press 1992. Tournament selection (default tournament size 7), subtree crossover (rate 0.9), subtree mutation (rate 0.05), raw MSE fitness, elitism = 1 best individual, max tree depth 17.
- **Parsimony GP** — adds a length penalty: fitness becomes `MSE + alpha * tree_size`. Reference: Poli & McPhee, "Parsimony Pressure Made Easy", GECCO 2008 ([proceedings](https://dl.acm.org/doi/10.1145/1389095.1389340)).
- **Lexicase GP** — Spector 2012; for symbolic regression typically ε-lexicase: La Cava, Spector, Danai, "ε-Lexicase Selection for Regression", GECCO 2016. Selects parents by filtering candidates on randomly ordered training cases, keeping only those within ε of the best on each case.

## Implementation Contract
The agent edits `gplearn/custom_sr.py` and provides four functions:

```python
def fitness_function(tree, X, y) -> float:
    # Lower is better.
    ...

def selection(population, fitnesses, n_select, tournament_size=7) -> list:
    # Return n_select selected individuals (copies).
    ...

def crossover(parent1, parent2, n_features, max_depth=17):
    # Return a new offspring expression tree.
    ...

def mutation(parent, n_features, max_depth=17):
    # Return a new mutated expression tree.
    ...

def evolve_one_generation(population, fitnesses, X_train, y_train,
                          n_features, pop_size,
                          crossover_rate=0.9, mutation_rate=0.05,
                          max_depth=17) -> list:
    # Return the next-generation population (length pop_size).
    ...
```

Available helpers from the skeleton: `safe_evaluate(tree, X)`, `generate_tree('grow'|'full', max_depth, n_features)`, `Tree.copy/size/depth/get_all_nodes()`. Reference code that may be read for context: `gplearn/gplearn/genetic.py`, `gplearn/gplearn/_program.py`, `gplearn/gplearn/fitness.py`. The output must remain an executable symbolic expression, not a black-box predictor.

## Fixed Pipeline & Evaluation
Your GP is evaluated on a held-out suite of standard symbolic-regression
benchmarks drawn from the usual families (low-dimensional univariate and
bivariate targets — polynomial, transcendental, and trigonometric). The exact
target functions are **not** disclosed: at run time the fixed driver hands your
search only sampled `(X_train, y_train)` training pairs and a set of test inputs
`X_test`. Your search must fit those samples and produce an executable
expression; the best evolved expression is evaluated on the held-out test inputs
by a host-side scorer.

Metric: **R²** on the held-out test set (higher is better; max 1.0). A train-set
fit summary (train R² and the discovered expression) is reported as feedback.
