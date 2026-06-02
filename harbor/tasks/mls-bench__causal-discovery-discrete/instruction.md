# MLS-Bench: causal-discovery-discrete

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
- `causal-bnlearn/bench/metrics.py`


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

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `pc` baseline — editable region  [READ-ONLY — reference implementation]

In `causal-bnlearn/bench/custom_algorithm.py`:

```python
Lines 3–32:
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
    13:     from causallearn.utils.PCUtils import SkeletonDiscovery, Meek, UCSepset
    14:     from causallearn.utils.cit import CIT
    15: 
    16:     alpha = 0.05
    17:     indep_test = CIT(X, "chisq")
    18: 
    19:     # Step 1: skeleton discovery via chi-squared CI tests (stable PC)
    20:     cg_1 = SkeletonDiscovery.skeleton_discovery(
    21:         X, alpha, indep_test, stable=True,
    22:         background_knowledge=None, verbose=False,
    23:         show_progress=False, node_names=None,
    24:     )
    25: 
    26:     # Step 2: orient unshielded colliders using UC-sepset rule (priority=2)
    27:     cg_2 = UCSepset.uc_sepset(cg_1, 2, background_knowledge=None)
    28: 
    29:     # Step 3: complete orientation with Meek rules
    30:     cg = Meek.meek(cg_2, background_knowledge=None)
    31: 
    32:     return cg.G
    33: # =====================================================================
```

### `ges` baseline — editable region  [READ-ONLY — reference implementation]

In `causal-bnlearn/bench/custom_algorithm.py`:

```python
Lines 3–20:
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
    13:     from causallearn.search.ScoreBased.GES import ges
    14: 
    15:     result = ges(
    16:         X,
    17:         score_func="local_score_BDeu",
    18:         parameters={"sample_prior": 1.0, "structure_prior": 1.0},
    19:     )
    20:     return result["G"]
    21: # =====================================================================
```

### `grasp` baseline — editable region  [READ-ONLY — reference implementation]

In `causal-bnlearn/bench/custom_algorithm.py`:

```python
Lines 3–21:
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
    13:     from causallearn.search.PermutationBased.GRaSP import grasp
    14: 
    15:     G = grasp(
    16:         X,
    17:         score_func="local_score_BDeu",
    18:         depth=3,
    19:         parameters={"sample_prior": 1.0, "structure_prior": 1.0},
    20:     )
    21:     return G
    22: # =====================================================================
```

### `boss` baseline — editable region  [READ-ONLY — reference implementation]

In `causal-bnlearn/bench/custom_algorithm.py`:

```python
Lines 3–20:
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
    13:     from causallearn.search.PermutationBased.BOSS import boss
    14: 
    15:     G = boss(
    16:         X,
    17:         score_func="local_score_BDeu",
    18:         parameters={"sample_prior": 1.0, "structure_prior": 1.0},
    19:     )
    20:     return G
    21: # =====================================================================
```

### `hc` baseline — editable region  [READ-ONLY — reference implementation]

In `causal-bnlearn/bench/custom_algorithm.py`:

```python
Lines 3–129:
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
    13:     from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
    14:     from causallearn.score.LocalScoreFunction import local_score_BDeu
    15:     from causallearn.utils.DAG2CPDAG import dag2cpdag
    16: 
    17:     N = X.shape[1]
    18:     # Pass parameters=None so local_score_BDeu auto-computes r_i_map from data
    19:     score_func = LocalScoreClass(
    20:         data=X, local_score_fun=local_score_BDeu, parameters=None
    21:     )
    22: 
    23:     nodes = [GraphNode(f"X{i + 1}") for i in range(N)]
    24:     adj = np.zeros((N, N), dtype=int)
    25: 
    26:     # Cache local scores (one per node)
    27:     local_scores = np.zeros(N)
    28:     for j in range(N):
    29:         local_scores[j] = score_func.score(j, [])
    30: 
    31:     def _has_path(src, tgt):
    32:         """DFS check: is there a directed path from src to tgt in adj?"""
    33:         visited = set()
    34:         stack = [src]
    35:         while stack:
    36:             node = stack.pop()
    37:             if node == tgt:
    38:                 return True
    39:             if node in visited:
    40:                 continue
    41:             visited.add(node)
    42:             for c in np.where(adj[node] == 1)[0]:
    43:                 if int(c) not in visited:
    44:                     stack.append(int(c))
    45:         return False
    46: 
    47:     # Greedy hill-climbing: add / delete / reverse
    48:     improved = True
    49:     while improved:
    50:         improved = False
    51:         best_delta = 0.0
    52:         best_op = None
    53: 
    54:         for i in range(N):
    55:             for j in range(N):
    56:                 if i == j:
    57:                     continue
    58: 
    59:                 if adj[i, j] == 0 and adj[j, i] == 0:
    60:                     # --- Try ADD i -> j (only if no cycle) ---
    61:                     if not _has_path(j, i):
    62:                         pj_new = sorted(
    63:                             np.where(adj[:, j] == 1)[0].tolist() + [i]
    64:                         )
    65:                         new_sj = score_func.score(j, pj_new)
    66:                         delta = new_sj - local_scores[j]
    67:                         if delta < best_delta - 1e-6:
    68:                             best_delta = delta
    69:                             best_op = ("add", i, j)
    70: 
    71:                 elif adj[i, j] == 1:
    72:                     # --- Try DELETE i -> j ---
    73:                     pj_new = [
    74:                         p for p in np.where(adj[:, j] == 1)[0] if p != i
    75:                     ]
    76:                     new_sj = score_func.score(j, sorted(pj_new))
    77:                     delta = new_sj - local_scores[j]
    78:                     if delta < best_delta - 1e-6:
    79:                         best_delta = delta
    80:                         best_op = ("delete", i, j)
    81: 
    82:                     # --- Try REVERSE i -> j  to  j -> i ---
    83:                     adj[i, j] = 0  # temporarily remove
    84:                     if not _has_path(i, j):
    85:                         pj_del = sorted(
    86:                             np.where(adj[:, j] == 1)[0].tolist()
    87:                         )
    88:                         new_sj = score_func.score(j, pj_del)
    89:                         pi_new = sorted(
    90:                             np.where(adj[:, i] == 1)[0].tolist() + [j]
    91:                         )
    92:                         new_si = score_func.score(i, pi_new)
    93:                         delta = (
    94:                             (new_sj - local_scores[j])
    95:                             + (new_si - local_scores[i])
    96:                         )
    97:                         if delta < best_delta - 1e-6:
    98:                             best_delta = delta
    99:                             best_op = ("reverse", i, j)
   100:                     adj[i, j] = 1  # restore
   101: 
   102:         if best_op is not None:
   103:             op_type, i, j = best_op
   104:             if op_type == "add":
   105:                 adj[i, j] = 1
   106:             elif op_type == "delete":
   107:                 adj[i, j] = 0
   108:             elif op_type == "reverse":
   109:                 adj[i, j] = 0
   110:                 adj[j, i] = 1
   111:             # Recompute affected local scores
   112:             local_scores[j] = score_func.score(
   113:                 j, sorted(np.where(adj[:, j] == 1)[0].tolist())
   114:             )
   115:             if op_type == "reverse":
   116:                 local_scores[i] = score_func.score(
   117:                     i, sorted(np.where(adj[:, i] == 1)[0].tolist())
   118:                 )
   119:             improved = True
   120: 
   121:     # Build GeneralGraph from learned DAG
   122:     G = GeneralGraph(nodes)
   123:     for i in range(N):
   124:         for j in range(N):
   125:             if adj[i, j] == 1:
   126:                 G.add_directed_edge(nodes[i], nodes[j])
   127: 
   128:     G = dag2cpdag(G)
   129:     return G
   130: # =====================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
