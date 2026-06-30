# MLS-Bench: causal-discovery-discrete

# Causal Discovery on Discrete Bayesian Network Datasets

## Research Question
Design a causal discovery algorithm that recovers the **CPDAG** (Completed
Partially Directed Acyclic Graph) from purely observational, integer-coded
discrete data sampled from real-world Bayesian networks.

## Background
Real-world Bayesian networks drawn from diverse domains (medicine, biology,
meteorology, insurance, agriculture, IT) have known ground-truth DAGs with
discrete variables and conditional probability tables.

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
(inclusive, 1-indexed)**. Edits that change code outside these ranges — or creating new files, or
deleting whole files — will cause your submission to be invalid.

The line numbers mark an editable **region**, not a fixed line-count budget: you
may add or remove lines inside it. Only code outside the editable ranges must
stay unchanged.

- `causal-bnlearn/bench/custom_algorithm.py`
- editable lines **3–14**


Other files you may **read** for context (do not modify):
- `causal-bnlearn/bench/run_eval.py`


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
Lines 3–27:
     1: import numpy as np
     2: from causallearn.graph.GeneralGraph import GeneralGraph
     3: def run_causal_discovery(X: np.ndarray) -> GeneralGraph:
     4:     """
     5:     Input:  X of shape (n_samples, n_variables), integer-encoded discrete data
     6:     Output: estimated CPDAG as causallearn.graph.GeneralGraph.GeneralGraph
     7:     """
     8:     from causallearn.utils.PCUtils import SkeletonDiscovery, Meek, UCSepset
     9:     from causallearn.utils.cit import CIT
    10: 
    11:     alpha = 0.05
    12:     indep_test = CIT(X, "chisq")
    13: 
    14:     # Step 1: skeleton discovery via chi-squared CI tests (stable PC)
    15:     cg_1 = SkeletonDiscovery.skeleton_discovery(
    16:         X, alpha, indep_test, stable=True,
    17:         background_knowledge=None, verbose=False,
    18:         show_progress=False, node_names=None,
    19:     )
    20: 
    21:     # Step 2: orient unshielded colliders using UC-sepset rule (priority=2)
    22:     cg_2 = UCSepset.uc_sepset(cg_1, 2, background_knowledge=None)
    23: 
    24:     # Step 3: complete orientation with Meek rules
    25:     cg = Meek.meek(cg_2, background_knowledge=None)
    26: 
    27:     return cg.G
    28: # =====================================================================
```

### `ges` baseline — editable region  [READ-ONLY — reference implementation]

In `causal-bnlearn/bench/custom_algorithm.py`:

```python
Lines 3–15:
     1: import numpy as np
     2: from causallearn.graph.GeneralGraph import GeneralGraph
     3: def run_causal_discovery(X: np.ndarray) -> GeneralGraph:
     4:     """
     5:     Input:  X of shape (n_samples, n_variables), integer-encoded discrete data
     6:     Output: estimated CPDAG as causallearn.graph.GeneralGraph.GeneralGraph
     7:     """
     8:     from causallearn.search.ScoreBased.GES import ges
     9: 
    10:     result = ges(
    11:         X,
    12:         score_func="local_score_BDeu",
    13:         parameters={"sample_prior": 1.0, "structure_prior": 1.0},
    14:     )
    15:     return result["G"]
    16: # =====================================================================
```

### `grasp` baseline — editable region  [READ-ONLY — reference implementation]

In `causal-bnlearn/bench/custom_algorithm.py`:

```python
Lines 3–16:
     1: import numpy as np
     2: from causallearn.graph.GeneralGraph import GeneralGraph
     3: def run_causal_discovery(X: np.ndarray) -> GeneralGraph:
     4:     """
     5:     Input:  X of shape (n_samples, n_variables), integer-encoded discrete data
     6:     Output: estimated CPDAG as causallearn.graph.GeneralGraph.GeneralGraph
     7:     """
     8:     from causallearn.search.PermutationBased.GRaSP import grasp
     9: 
    10:     G = grasp(
    11:         X,
    12:         score_func="local_score_BDeu",
    13:         depth=3,
    14:         parameters={"sample_prior": 1.0, "structure_prior": 1.0},
    15:     )
    16:     return G
    17: # =====================================================================
```

### `boss` baseline — editable region  [READ-ONLY — reference implementation]

In `causal-bnlearn/bench/custom_algorithm.py`:

```python
Lines 3–15:
     1: import numpy as np
     2: from causallearn.graph.GeneralGraph import GeneralGraph
     3: def run_causal_discovery(X: np.ndarray) -> GeneralGraph:
     4:     """
     5:     Input:  X of shape (n_samples, n_variables), integer-encoded discrete data
     6:     Output: estimated CPDAG as causallearn.graph.GeneralGraph.GeneralGraph
     7:     """
     8:     from causallearn.search.PermutationBased.BOSS import boss
     9: 
    10:     G = boss(
    11:         X,
    12:         score_func="local_score_BDeu",
    13:         parameters={"sample_prior": 1.0, "structure_prior": 1.0},
    14:     )
    15:     return G
    16: # =====================================================================
```

### `hc` baseline — editable region  [READ-ONLY — reference implementation]

In `causal-bnlearn/bench/custom_algorithm.py`:

```python
Lines 3–124:
     1: import numpy as np
     2: from causallearn.graph.GeneralGraph import GeneralGraph
     3: def run_causal_discovery(X: np.ndarray) -> GeneralGraph:
     4:     """
     5:     Input:  X of shape (n_samples, n_variables), integer-encoded discrete data
     6:     Output: estimated CPDAG as causallearn.graph.GeneralGraph.GeneralGraph
     7:     """
     8:     from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
     9:     from causallearn.score.LocalScoreFunction import local_score_BDeu
    10:     from causallearn.utils.DAG2CPDAG import dag2cpdag
    11: 
    12:     N = X.shape[1]
    13:     # Pass parameters=None so local_score_BDeu auto-computes r_i_map from data
    14:     score_func = LocalScoreClass(
    15:         data=X, local_score_fun=local_score_BDeu, parameters=None
    16:     )
    17: 
    18:     nodes = [GraphNode(f"X{i + 1}") for i in range(N)]
    19:     adj = np.zeros((N, N), dtype=int)
    20: 
    21:     # Cache local scores (one per node)
    22:     local_scores = np.zeros(N)
    23:     for j in range(N):
    24:         local_scores[j] = score_func.score(j, [])
    25: 
    26:     def _has_path(src, tgt):
    27:         """DFS check: is there a directed path from src to tgt in adj?"""
    28:         visited = set()
    29:         stack = [src]
    30:         while stack:
    31:             node = stack.pop()
    32:             if node == tgt:
    33:                 return True
    34:             if node in visited:
    35:                 continue
    36:             visited.add(node)
    37:             for c in np.where(adj[node] == 1)[0]:
    38:                 if int(c) not in visited:
    39:                     stack.append(int(c))
    40:         return False
    41: 
    42:     # Greedy hill-climbing: add / delete / reverse
    43:     improved = True
    44:     while improved:
    45:         improved = False
    46:         best_delta = 0.0
    47:         best_op = None
    48: 
    49:         for i in range(N):
    50:             for j in range(N):
    51:                 if i == j:
    52:                     continue
    53: 
    54:                 if adj[i, j] == 0 and adj[j, i] == 0:
    55:                     # --- Try ADD i -> j (only if no cycle) ---
    56:                     if not _has_path(j, i):
    57:                         pj_new = sorted(
    58:                             np.where(adj[:, j] == 1)[0].tolist() + [i]
    59:                         )
    60:                         new_sj = score_func.score(j, pj_new)
    61:                         delta = new_sj - local_scores[j]
    62:                         if delta < best_delta - 1e-6:
    63:                             best_delta = delta
    64:                             best_op = ("add", i, j)
    65: 
    66:                 elif adj[i, j] == 1:
    67:                     # --- Try DELETE i -> j ---
    68:                     pj_new = [
    69:                         p for p in np.where(adj[:, j] == 1)[0] if p != i
    70:                     ]
    71:                     new_sj = score_func.score(j, sorted(pj_new))
    72:                     delta = new_sj - local_scores[j]
    73:                     if delta < best_delta - 1e-6:
    74:                         best_delta = delta
    75:                         best_op = ("delete", i, j)
    76: 
    77:                     # --- Try REVERSE i -> j  to  j -> i ---
    78:                     adj[i, j] = 0  # temporarily remove
    79:                     if not _has_path(i, j):
    80:                         pj_del = sorted(
    81:                             np.where(adj[:, j] == 1)[0].tolist()
    82:                         )
    83:                         new_sj = score_func.score(j, pj_del)
    84:                         pi_new = sorted(
    85:                             np.where(adj[:, i] == 1)[0].tolist() + [j]
    86:                         )
    87:                         new_si = score_func.score(i, pi_new)
    88:                         delta = (
    89:                             (new_sj - local_scores[j])
    90:                             + (new_si - local_scores[i])
    91:                         )
    92:                         if delta < best_delta - 1e-6:
    93:                             best_delta = delta
    94:                             best_op = ("reverse", i, j)
    95:                     adj[i, j] = 1  # restore
    96: 
    97:         if best_op is not None:
    98:             op_type, i, j = best_op
    99:             if op_type == "add":
   100:                 adj[i, j] = 1
   101:             elif op_type == "delete":
   102:                 adj[i, j] = 0
   103:             elif op_type == "reverse":
   104:                 adj[i, j] = 0
   105:                 adj[j, i] = 1
   106:             # Recompute affected local scores
   107:             local_scores[j] = score_func.score(
   108:                 j, sorted(np.where(adj[:, j] == 1)[0].tolist())
   109:             )
   110:             if op_type == "reverse":
   111:                 local_scores[i] = score_func.score(
   112:                     i, sorted(np.where(adj[:, i] == 1)[0].tolist())
   113:                 )
   114:             improved = True
   115: 
   116:     # Build GeneralGraph from learned DAG
   117:     G = GeneralGraph(nodes)
   118:     for i in range(N):
   119:         for j in range(N):
   120:             if adj[i, j] == 1:
   121:                 G.add_directed_edge(nodes[i], nodes[j])
   122: 
   123:     G = dag2cpdag(G)
   124:     return G
   125: # =====================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
