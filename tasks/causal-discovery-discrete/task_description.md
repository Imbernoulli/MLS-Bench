# Causal Discovery on Discrete Bayesian Network Datasets (bnlearn)

## Research Question
Design a causal discovery algorithm that recovers the **CPDAG** (Completed
Partially Directed Acyclic Graph) from purely observational, integer-coded
discrete data sampled from real-world Bayesian networks in the bnlearn
repository.

## Background
The bnlearn repository (https://www.bnlearn.com/bnrepository/) hosts a
collection of well-known Bayesian network benchmarks from diverse domains
(medicine, biology, meteorology, insurance, agriculture, IT). Each network has
a known ground-truth DAG with discrete variables and conditional probability
tables.

Under the faithfulness assumption, observational data can identify only the
Markov Equivalence Class (MEC) of the true DAG, represented by a CPDAG. The
challenge lies in handling discrete data with varying cardinalities, network
sizes (small to >70 nodes), and edge densities, without over-specializing to a
single scale or cardinality pattern.

## Task
Implement a causal discovery algorithm in `causal-bnlearn/bench/custom_algorithm.py`. The
`run_causal_discovery(X)` function receives integer-encoded discrete
observational data and must return the estimated CPDAG as a
`causallearn.graph.GeneralGraph.GeneralGraph` object.

```python
def run_causal_discovery(X: np.ndarray) -> GeneralGraph:
    """
    Input:  X of shape (n_samples, n_variables), integer-encoded discrete data
    Output: estimated CPDAG as causallearn.graph.GeneralGraph.GeneralGraph
    """
```

## Evaluation Networks

| Label      | Nodes | Edges | Domain                       |
|------------|-------|-------|------------------------------|
| Cancer     | 5     | 4     | Medical                      |
| Child      | 20    | 25    | Medical                      |
| Alarm      | 37    | 46    | Medical monitoring           |
| Hailfinder | 56    | 66    | Meteorology                  |
| Win95pts   | 76    | 112   | IT (Windows troubleshooting) |

Each network is sampled with a fixed observational sample size; the agent must
generalize across small/medium/large networks and across different cardinality
patterns.

## Metrics
Metrics are computed between the estimated CPDAG and the ground-truth CPDAG
(converted from the true DAG via `dag2cpdag`):
- **SHD** (Structural Hamming Distance): total edge errors (lower is better)
- **Adjacency Precision / Recall**: skeleton recovery quality (higher is better)
- **Arrow Precision / Recall**: edge orientation accuracy (higher is better)

## Reference baselines
The benchmark ships several classical baselines for comparison. Citations are
provided so the agent can study the prior art; default hyperparameters are the
ones recommended in the cited papers (e.g., chi-squared CI test for PC, BDeu
score for the score-based methods).

- `pc`: Peter-Clark algorithm with chi-squared CI test. Constraint-based.
  Spirtes, Glymour & Scheines, *Causation, Prediction, and Search* (MIT Press,
  2nd ed., 2000).
- `ges`: Greedy Equivalence Search with BDeu score. Score-based. Chickering,
  "Optimal Structure Identification With Greedy Search," JMLR 3, 2002.
- `grasp`: Greedy Relaxations of the Sparsest Permutation with BDeu score.
  Permutation-based. Lam, Andrews & Ramsey, "Greedy Relaxations of the Sparsest
  Permutation Algorithm," UAI 2022 (arXiv:2206.05421).
- `boss`: Best Order Score Search with BDeu score. Permutation-based. Andrews
  et al., "Fast Scalable and Accurate Discovery of DAGs Using the Best Order
  Score Search and Grow-Shrink Trees," NeurIPS 2023 (arXiv:2310.17679).
- `hc`: Hill-Climbing search with BDeu score. Score-based, classical local
  search baseline.

The contribution should be a modular causal discovery procedure for discrete
observational data, such as a constraint-based, score-based, permutation-based,
hybrid, or otherwise principled alternative, while staying within the provided
causal graph interface.
