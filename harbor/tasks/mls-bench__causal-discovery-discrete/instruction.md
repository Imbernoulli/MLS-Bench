# MLS-Bench: causal-discovery-discrete

# Causal Discovery on Discrete Observational Data

## Research Question
Design a causal discovery algorithm that recovers the **CPDAG** (Completed
Partially Directed Acyclic Graph) from purely observational, integer-coded
discrete data sampled from real-world Bayesian networks.

## Background
Under the faithfulness assumption, observational data can identify only the
Markov Equivalence Class (MEC) of the true DAG, represented by a CPDAG. The
challenge lies in handling discrete data with varying cardinalities, network
sizes, and edge densities, without over-specializing to a single scale or
cardinality pattern.

## Task
Implement a causal discovery algorithm in `bench/custom_algorithm.py`. The
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

The contribution should be a modular causal discovery procedure for discrete
observational data, such as a constraint-based, score-based, permutation-based,
hybrid, or otherwise principled alternative, while staying within the provided
causal graph interface.

## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/causal-bnlearn/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `causal-bnlearn/bench/custom_algorithm.py`
- editable lines **3–14**

Other files you may **read** for context (do not modify):
- `causal-bnlearn/bench/run_eval.py`
- `causal-bnlearn/bench/data_gen.py`

## Readable Context

### `causal-bnlearn/bench/custom_algorithm.py`  [EDITABLE — lines 3–14 only]

```python
     1: import numpy as np
     2: from causallearn.graph.GeneralGraph import GeneralGraph
     3: from causallearn.graph.GraphNode import GraphNode
     4:
     5: # =====================================================================
     6: # EDITABLE: implement run_causal_discovery below
     7: # =====================================================================
     8: def run_causal_discovery(X: np.ndarray) -> GeneralGraph:
     9:     """
    10:     Input:  X of shape (n_samples, n_variables), integer-encoded discrete data
    11:     Output: estimated CPDAG as causallearn.graph.GeneralGraph.GeneralGraph
    12:     """
    13:     nodes = [GraphNode(f"X{i + 1}") for i in range(X.shape[1])]
    14:     return GeneralGraph(nodes)
    15: # =====================================================================
```

## Tips

- Keep the function/class signatures of the editable regions identical;
  the runtime imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.

Good luck.
