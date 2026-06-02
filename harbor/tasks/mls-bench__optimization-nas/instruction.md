# MLS-Bench: optimization-nas

# Sample-Efficient Neural Architecture Search

## Objective
Design and implement a novel **sample-efficient** NAS optimizer that discovers high-performing architectures in the NAS-Bench-201 search space under a **strict query budget**. Your code goes in the `NASOptimizer` class in `custom_nas_search.py`. Three reference implementations (Random Search, REA, and a BANANAS-style predictor-guided search) are provided as read-only.

## Research Question
With only a **strict architecture evaluation budget**, how can a search strategy maximize the expected accuracy of the best-found architecture?

This is the regime in which real-world NAS is actually hard: the full benchmark contains 15,625 architectures, but the agent can only query a small fraction of them, so naïve enumeration is impossible and algorithmic differences are load-bearing. Sample-efficient NAS has been studied by BANANAS (White, Neiswanger, and Savani, AAAI 2021; arXiv:1910.11858), NPENAS (Wei, Niu, Chen, and Wang, IEEE TNNLS, 2022), and NAS-Bench-Suite (White et al., 2022) and consistently shows a measurable gap between random search, regularized evolution, and predictor-guided methods at low query budgets.

## Search Space
- NAS-Bench-201 cell: 4 nodes, 6 edges, 5 operations per edge (Dong and Yang, "NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search", ICLR 2020; arXiv:2001.00326).
- Operations: `skip_connect, none, nor_conv_3x3, nor_conv_1x1, avg_pool_3x3`.
- 5^6 = 15,625 architectures total.
- An architecture is represented as a list of 6 integers in `[0, 4]`.

## Evaluation Protocol
- **Query budget: `NAS_EPOCHS`** validation queries per run (the harness enforces this; exceeding it aborts the run).
- After search, the harness performs one final unbudgeted test query on your returned architecture; this does not count against your budget.

## What Counts as a Contribution
Acceptable research directions (this list is not exhaustive):
- **Better acquisition functions**: e.g. UCB / EI over a learned predictor, Thompson sampling, information-theoretic criteria.
- **Better surrogate models**: GPs on path-encoded architectures, GNN predictors, MLP ensembles, zero-cost proxy hybrids (Mellor, Turner, Storkey, and Crowley, "Neural Architecture Search without Training", ICML 2021; Abdelfattah, Mehrotra, Dudziak, and Lane, "Zero-Cost Proxies for Lightweight NAS", ICLR 2021).
- **Smarter exploration–exploitation mixing**: local search around the Pareto front, portfolio methods, warm-started evolution.
- **Encoding choices**: adjacency vs path encoding (White, Neiswanger, Nolen, and Savani, "A Study on Encodings for Neural Architecture Search", NeurIPS 2020 showed path encoding substantially improves predictor accuracy at low K).

What does **not** count:
- Increasing the effective budget (e.g. re-querying the same architecture, wrapping queries, etc.). The harness counts every call to `api.query_val_accuracy` and will terminate when the budget is exhausted.
- Hard-coding known good architectures from NAS-Bench-201 literature.

## Baselines (paper-cited reference implementations)

| Name | Strategy |
|------|----------|
| `random_search` | Uniform sampling over valid architectures. |
| `rea` | Regularized Evolution (Real, Aggarwal, Huang, and Le, AAAI 2019; arXiv:1802.01548) with tournament selection (paper-default `S = 10`, `population_size = 20`) and 1-edge mutation. |
| `bananas` | Predictor-guided: MLP ensemble over path encodings, pick candidate with highest predicted val_acc (White, Neiswanger, and Savani, AAAI 2021; arXiv:1910.11858). Paper-default 5-MLP ensemble, 100 mutation candidates per acquisition. |


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/naslib/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `naslib/custom_nas_search.py`
- editable lines **163–234**




## Readable Context


### `naslib/custom_nas_search.py`  [EDITABLE — lines 163–234 only]

```python
     1: # Custom NAS optimizer for MLS-Bench (NAS-Bench-201, sample-efficient regime)
     2: #
     3: # EDITABLE section: NASOptimizer class — implement your search strategy.
     4: # FIXED sections: everything else (search space, benchmark API, evaluation loop).
     5: #
     6: # The NAS-Bench-201 search space has 15625 architectures (5 ops, 6 edges).
     7: # Evaluation is tabular — query the benchmark for any architecture's accuracy.
     8: # No actual neural network training is needed.
     9: #
    10: # IMPORTANT: You have a STRICT budget of NAS_EPOCHS validation queries
    11: # (default 30). The BenchmarkAPI enforces this and will raise
    12: # BudgetExceededError if you exceed it. One final test query at the end is
    13: # free and not counted against the budget.
    14: import os
    15: import sys
    16: import time
    17: import random
    18: import pickle
    19: import copy
    20: import numpy as np
    21: from pathlib import Path
    22: 
    23: 
    24: # =====================================================================
    25: # FIXED: NAS-Bench-201 Search Space Definition
    26: # =====================================================================
    27: NUM_EDGES = 6
    28: NUM_OPS = 5
    29: OP_NAMES = ["skip_connect", "none", "nor_conv_3x3", "nor_conv_1x1", "avg_pool_3x3"]
    30: 
    31: # Edge list: (source, target) for the 4-node cell
    32: # Node 0: input, Nodes 1-2: intermediate, Node 3: output
    33: EDGE_LIST = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
    34: 
    35: # Dataset name mapping for the benchmark lookup
    36: DATASET_MAP = {
    37:     "cifar10": "cifar10",
    38:     "cifar100": "cifar100",
    39:     "imagenet16": "ImageNet16-120",
    40: }
    41: 
    42: 
    43: class BudgetExceededError(RuntimeError):
    44:     """Raised when the validation query budget is exhausted."""
    45: 
    46: 
    47: def op_indices_to_arch_str(op_indices):
    48:     """Convert a list of 6 op indices to the NAS-Bench-201 architecture string."""
    49:     edge_op_dict = {
    50:         edge: OP_NAMES[op] for edge, op in zip(EDGE_LIST, op_indices)
    51:     }
    52:     op_edge_list = [
    53:         "{}~{}".format(edge_op_dict[(i, j)], i - 1)
    54:         for i, j in sorted(edge_op_dict, key=lambda x: x[1])
    55:     ]
    56:     return "|{}|+|{}|{}|+|{}|{}|{}|".format(*op_edge_list)
    57: 
    58: 
    59: def is_valid_arch(op_indices):
    60:     """Check architecture validity (not all-zero on any path)."""
    61:     # none=1 in OP_NAMES; reject if all edges to node 3 are 'none'
    62:     # or all edges from node 1 are 'none'
    63:     return not ((op_indices[0] == op_indices[1] == op_indices[2] == 1) or
    64:                 (op_indices[2] == op_indices[4] == op_indices[5] == 1))
    65: 
    66: 
    67: def random_architecture():
    68:     """Sample a random valid architecture as a list of 6 op indices."""
    69:     while True:
    70:         op_indices = [random.randint(0, NUM_OPS - 1) for _ in range(NUM_EDGES)]
    71:         if is_valid_arch(op_indices):
    72:             return op_indices
    73: 
    74: 
    75: def mutate_architecture(parent_op_indices):
    76:     """Mutate one random edge of the parent architecture."""
    77:     op_indices = list(parent_op_indices)
    78:     edge = random.randint(0, NUM_EDGES - 1)
    79:     available = [o for o in range(NUM_OPS) if o != parent_op_indices[edge]]
    80:     op_indices[edge] = random.choice(available)
    81:     return op_indices
    82: 
    83: 
    84: def get_neighbors(op_indices):
    85:     """Get all 1-edit-distance neighbors of an architecture."""
    86:     neighbors = []
    87:     for edge in range(NUM_EDGES):
    88:         for op in range(NUM_OPS):
    89:             if op != op_indices[edge]:
    90:                 nbr = list(op_indices)
    91:                 nbr[edge] = op
    92:                 neighbors.append(nbr)
    93:     return neighbors
    94: 
    95: 
    96: def path_encoding(op_indices):
    97:     """Path encoding of a NAS-Bench-201 cell (White et al., 2020).
    98: 
    99:     Enumerates every op-labeled path from input to output and returns a binary
   100:     indicator vector of length NUM_OPS**3 + NUM_OPS**2 + NUM_OPS (paths of
   101:     length 1, 2, 3 respectively). Useful as input to predictor models.
   102:     """
   103:     # Edges: 0:(1,2) 1:(1,3) 2:(1,4) 3:(2,3) 4:(2,4) 5:(3,4)
   104:     # i.e. from input(node 1) to output(node 4)
   105:     o = op_indices
   106:     enc_len = NUM_OPS ** 3 + NUM_OPS ** 2 + NUM_OPS
   107:     v = np.zeros(enc_len, dtype=np.float32)
   108:     # length-1 paths (direct 1->4)
   109:     v[o[2]] = 1.0
   110:     # length-2 paths (1->2->4, 1->3->4)
   111:     offset = NUM_OPS
   112:     v[offset + o[0] * NUM_OPS + o[4]] = 1.0
   113:     v[offset + o[1] * NUM_OPS + o[5]] = 1.0
   114:     # length-3 paths (1->2->3->4)
   115:     offset = NUM_OPS + NUM_OPS ** 2
   116:     v[offset + o[0] * NUM_OPS ** 2 + o[3] * NUM_OPS + o[5]] = 1.0
   117:     return v
   118: 
   119: 
   120: class BenchmarkAPI:
   121:     """Wrapper for querying NAS-Bench-201 with a hard validation-query budget."""
   122: 
   123:     def __init__(self, data, dataset_key, query_budget):
   124:         self.data = data
   125:         self.dataset_key = dataset_key
   126:         self.query_budget = int(query_budget)
   127:         self.query_count = 0
   128:         self._cache = {}  # repeated queries don't cost extra but still count
   129: 
   130:     @property
   131:     def remaining_budget(self):
   132:         return max(0, self.query_budget - self.query_count)
   133: 
   134:     def query_val_accuracy(self, op_indices):
   135:         """Query validation accuracy (counts against the budget).
   136: 
   137:         For cifar10, validation accuracy is from the 'cifar10-valid' split.
   138:         For cifar100 and ImageNet16-120, validation accuracy uses 'eval_acc1es'
   139:         from the respective split (standard NAS-Bench-201 search protocol).
   140:         """
   141:         if self.query_count >= self.query_budget:
   142:             raise BudgetExceededError(
   143:                 f"Validation query budget of {self.query_budget} exhausted."
   144:             )
   145:         self.query_count += 1
   146:         arch_str = op_indices_to_arch_str(op_indices)
   147:         if self.dataset_key == "cifar10":
   148:             return self.data[arch_str]["cifar10-valid"]["eval_acc1es"]
   149:         else:
   150:             return self.data[arch_str][self.dataset_key]["eval_acc1es"]
   151: 
   152:     # --- Harness-only methods (not counted against the agent's budget) ---
   153: 
   154:     def _query_test_accuracy_unbudgeted(self, op_indices):
   155:         """Query final test accuracy — only called by the harness after search."""
   156:         arch_str = op_indices_to_arch_str(op_indices)
   157:         return self.data[arch_str][self.dataset_key]["eval_acc1es"]
   158: 
   159: 
   160: # =====================================================================
   161: # EDITABLE: NAS Optimizer — implement your search strategy here
   162: # =====================================================================
   163: class NASOptimizer:
   164:     """Sample-efficient NAS search strategy.
   165: 
   166:     Implement a search algorithm that maximizes the test accuracy of the
   167:     best-found architecture under a STRICT validation-query budget
   168:     (self.num_epochs, default 30).
   169: 
   170:     The search space has 15625 architectures (5 ops x 6 edges). Each
   171:     architecture is a list of 6 integers in [0, 4].
   172: 
   173:     Available helper functions (defined above, fixed):
   174:         random_architecture()                  -> list[int]  (random valid arch)
   175:         mutate_architecture(parent)            -> list[int]  (1-edge mutation)
   176:         get_neighbors(op_indices)              -> list[list[int]]  (all 1-edit neighbors)
   177:         is_valid_arch(op_indices)              -> bool
   178:         op_indices_to_arch_str(op_indices)     -> str
   179:         path_encoding(op_indices)              -> np.ndarray (features for predictors)
   180: 
   181:     The benchmark API (self.api) provides ONE budgeted method:
   182:         api.query_val_accuracy(op_indices)     -> float   (costs 1 query)
   183:         api.query_count                        -> int     (queries used so far)
   184:         api.remaining_budget                   -> int     (queries left)
   185: 
   186:     The harness will call search_step(epoch) up to self.num_epochs times.
   187:     After each step, you should maintain self.best_arch so that
   188:     get_best_architecture() returns the architecture you most want the
   189:     harness to finally test (on the unbudgeted test split).
   190:     """
   191: 
   192:     def __init__(self, api, num_epochs, seed):
   193:         """Initialize the optimizer.
   194: 
   195:         Args:
   196:             api: BenchmarkAPI (with budget = num_epochs validation queries).
   197:             num_epochs: Total number of allowed validation queries (budget).
   198:             seed: Random seed for reproducibility.
   199:         """
   200:         self.api = api
   201:         self.num_epochs = num_epochs
   202:         self.seed = seed
   203: 
   204:         # TODO: Initialize your search state here
   205:         self.best_arch = None
   206:         self.best_val_acc = -1.0
   207: 
   208:     def search_step(self, epoch):
   209:         """Run one step of the search algorithm.
   210: 
   211:         Args:
   212:             epoch: Current search iteration (0-indexed)
   213: 
   214:         Returns:
   215:             dict: Metrics to log, must include 'best_val_acc' and 'queries'.
   216:         """
   217:         # Placeholder: random search (replace with your algorithm)
   218:         arch = random_architecture()
   219:         val_acc = self.api.query_val_accuracy(arch)
   220: 
   221:         if val_acc > self.best_val_acc:
   222:             self.best_val_acc = val_acc
   223:             self.best_arch = arch
   224: 
   225:         return {
   226:             "best_val_acc": self.best_val_acc,
   227:             "queries": self.api.query_count,
   228:             "current_val_acc": val_acc,
   229:         }
   230: 
   231:     def get_best_architecture(self):
   232:         """Return the architecture the harness will test (unbudgeted)."""
   233:         return self.best_arch
   234: 
   235: 
   236: # =====================================================================
   237: # FIXED: Main entry point — search + evaluation
   238: # =====================================================================
   239: if __name__ == "__main__":
   240:     # ── Configuration from environment ──
   241:     seed = int(os.environ.get("SEED", 42))
   242:     output_dir = os.environ.get("OUTPUT_DIR", "/tmp/nas_output")
   243:     env_name = os.environ.get("ENV", "cifar10")
   244:     num_epochs = int(os.environ.get("NAS_EPOCHS", 30))  # sample-efficient: K=30
   245: 
   246:     os.makedirs(output_dir, exist_ok=True)
   247: 
   248:     # ── Seeding ──
   249:     random.seed(seed)
   250:     np.random.seed(seed)
   251: 
   252:     # ── Map environment name to dataset key ──
   253:     dataset_key = DATASET_MAP.get(env_name)
   254:     if dataset_key is None:
   255:         print(f"ERROR: Unknown environment '{env_name}'. Must be one of: {list(DATASET_MAP.keys())}")
   256:         sys.exit(1)
   257: 
   258:     # ── Load NAS-Bench-201 benchmark data ──
   259:     _candidates = [
   260:         Path("/workspace/naslib/naslib/data/nb201_all.pickle"),
   261:         Path(__file__).resolve().parent / "naslib" / "data" / "nb201_all.pickle",
   262:         Path(__file__).resolve().parent / "data" / "nb201_all.pickle",
   263:         Path("naslib/data/nb201_all.pickle"),
   264:         Path("data/nb201_all.pickle"),
   265:     ]
   266:     data_path = None
   267:     for _p in _candidates:
   268:         if _p.exists():
   269:             data_path = _p
   270:             break
   271:     if data_path is None:
   272:         print(f"ERROR: Benchmark data not found. Searched: {[str(p) for p in _candidates]}")
   273:         sys.exit(1)
   274: 
   275:     print(f"Loading NAS-Bench-201 data from {data_path}...", flush=True)
   276:     with open(data_path, "rb") as f:
   277:         nb201_data = pickle.load(f)
   278:     print(f"Loaded {len(nb201_data)} architectures.", flush=True)
   279: 
   280:     # ── Create benchmark API with strict budget ──
   281:     api = BenchmarkAPI(nb201_data, dataset_key, query_budget=num_epochs)
   282: 
   283:     # ── Run search ──
   284:     print(f"Starting sample-efficient NAS on {env_name} (dataset={dataset_key}) "
   285:           f"with budget={num_epochs} queries, seed={seed}", flush=True)
   286: 
   287:     optimizer = NASOptimizer(api, num_epochs, seed)
   288: 
   289:     start_time = time.time()
   290:     for epoch in range(num_epochs):
   291:         if api.remaining_budget <= 0:
   292:             print(f"Budget exhausted at epoch {epoch}; stopping search.", flush=True)
   293:             break
   294:         try:
   295:             metrics = optimizer.search_step(epoch)
   296:         except BudgetExceededError as e:
   297:             print(f"BUDGET EXCEEDED at epoch {epoch}: {e}", flush=True)
   298:             break
   299: 
   300:         # Log training metrics every step (K=30 is small)
   301:         elapsed = time.time() - start_time
   302:         metrics_str = " ".join(
   303:             f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
   304:             for k, v in metrics.items()
   305:         )
   306:         print(f"TRAIN_METRICS epoch={epoch+1} {metrics_str} "
   307:               f"elapsed={elapsed:.1f}s", flush=True)
   308: 
   309:     # ── Final evaluation (UNBUDGETED) ──
   310:     best_arch = optimizer.get_best_architecture()
   311:     if best_arch is None:
   312:         print("ERROR: No architecture found during search!", flush=True)
   313:         sys.exit(1)
   314:     if not is_valid_arch(best_arch):
   315:         print(f"ERROR: Returned architecture {best_arch} is invalid.", flush=True)
   316:         sys.exit(1)
   317: 
   318:     best_arch_str = op_indices_to_arch_str(best_arch)
   319:     test_acc = api._query_test_accuracy_unbudgeted(best_arch)
   320:     total_queries = api.query_count
   321:     total_time = time.time() - start_time
   322: 
   323:     print(f"\n{'='*60}", flush=True)
   324:     print(f"Search complete on {env_name} (dataset={dataset_key})", flush=True)
   325:     print(f"Best architecture: {best_arch} -> {best_arch_str}", flush=True)
   326:     print(f"Total val queries used: {total_queries} / {num_epochs}", flush=True)
   327:     print(f"Total time: {total_time:.1f}s", flush=True)
   328:     print(f"{'='*60}", flush=True)
   329: 
   330:     # Output test metric for parser
   331:     print(f"TEST_METRICS test_accuracy={test_acc:.4f}", flush=True)
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `random_search` baseline — editable region  [READ-ONLY — reference implementation]

In `naslib/custom_nas_search.py`:

```python
Lines 163–188:
   160: # =====================================================================
   161: # EDITABLE: NAS Optimizer — implement your search strategy here
   162: # =====================================================================
   163: class NASOptimizer:
   164:     """Random Search — uniformly sample architectures and track the best."""
   165: 
   166:     def __init__(self, api, num_epochs, seed):
   167:         self.api = api
   168:         self.num_epochs = num_epochs
   169:         self.seed = seed
   170:         self.best_arch = None
   171:         self.best_val_acc = -1.0
   172: 
   173:     def search_step(self, epoch):
   174:         arch = random_architecture()
   175:         val_acc = self.api.query_val_accuracy(arch)
   176: 
   177:         if val_acc > self.best_val_acc:
   178:             self.best_val_acc = val_acc
   179:             self.best_arch = arch
   180: 
   181:         return {
   182:             "best_val_acc": self.best_val_acc,
   183:             "queries": self.api.query_count,
   184:             "current_val_acc": val_acc,
   185:         }
   186: 
   187:     def get_best_architecture(self):
   188:         return self.best_arch
   189: 
   190: # =====================================================================
   191: # FIXED: Main entry point — search + evaluation
```

### `rea` baseline — editable region  [READ-ONLY — reference implementation]

In `naslib/custom_nas_search.py`:

```python
Lines 163–219:
   160: # =====================================================================
   161: # EDITABLE: NAS Optimizer — implement your search strategy here
   162: # =====================================================================
   163: class NASOptimizer:
   164:     """REA — Regularized Evolution Algorithm for NAS (low-budget variant).
   165: 
   166:     Population size 10 and tournament size 3, tuned for K=30 queries
   167:     following NAS-Bench-Suite (White et al., 2022) low-budget recipes.
   168:     """
   169: 
   170:     def __init__(self, api, num_epochs, seed):
   171:         self.api = api
   172:         self.num_epochs = num_epochs
   173:         self.seed = seed
   174: 
   175:         self.population_size = 10
   176:         self.tournament_size = 3
   177:         self.population = []  # list of (arch, val_acc)
   178:         self.best_arch = None
   179:         self.best_val_acc = -1.0
   180: 
   181:     def _update_best(self, arch, val_acc):
   182:         if val_acc > self.best_val_acc:
   183:             self.best_val_acc = val_acc
   184:             self.best_arch = list(arch)
   185: 
   186:     def search_step(self, epoch):
   187:         if epoch < self.population_size:
   188:             # Seed initial population with random architectures
   189:             arch = random_architecture()
   190:             val_acc = self.api.query_val_accuracy(arch)
   191:             self.population.append((arch, val_acc))
   192:         else:
   193:             # Tournament selection
   194:             k = min(self.tournament_size, len(self.population))
   195:             sample_indices = random.sample(range(len(self.population)), k)
   196:             parent_idx = max(sample_indices, key=lambda i: self.population[i][1])
   197:             parent_arch = self.population[parent_idx][0]
   198: 
   199:             # Mutation
   200:             child_arch = mutate_architecture(parent_arch)
   201:             while not is_valid_arch(child_arch):
   202:                 child_arch = mutate_architecture(parent_arch)
   203:             child_val_acc = self.api.query_val_accuracy(child_arch)
   204: 
   205:             # Add child and remove oldest (regularization)
   206:             self.population.append((child_arch, child_val_acc))
   207:             self.population.pop(0)
   208:             arch, val_acc = child_arch, child_val_acc
   209: 
   210:         self._update_best(arch, val_acc)
   211: 
   212:         return {
   213:             "best_val_acc": self.best_val_acc,
   214:             "queries": self.api.query_count,
   215:             "population_size": len(self.population),
   216:         }
   217: 
   218:     def get_best_architecture(self):
   219:         return self.best_arch
   220: 
   221: # =====================================================================
   222: # FIXED: Main entry point — search + evaluation
```

### `bananas` baseline — editable region  [READ-ONLY — reference implementation]

In `naslib/custom_nas_search.py`:

```python
Lines 163–282:
   160: # =====================================================================
   161: # EDITABLE: NAS Optimizer — implement your search strategy here
   162: # =====================================================================
   163: class _TinyMLP:
   164:     """2-layer numpy MLP regressor trained with Adam + MSE."""
   165: 
   166:     def __init__(self, in_dim, hidden=64, seed=0):
   167:         rs = np.random.RandomState(seed)
   168:         self.W1 = rs.randn(in_dim, hidden).astype(np.float32) * (1.0 / np.sqrt(in_dim))
   169:         self.b1 = np.zeros(hidden, dtype=np.float32)
   170:         self.W2 = rs.randn(hidden, 1).astype(np.float32) * (1.0 / np.sqrt(hidden))
   171:         self.b2 = np.zeros(1, dtype=np.float32)
   172: 
   173:     @staticmethod
   174:     def _relu(x):
   175:         return np.maximum(x, 0.0)
   176: 
   177:     def forward(self, X):
   178:         self._X = X
   179:         self._z1 = X @ self.W1 + self.b1
   180:         self._a1 = self._relu(self._z1)
   181:         return (self._a1 @ self.W2 + self.b2).squeeze(-1)
   182: 
   183:     def fit(self, X, y, epochs=200, lr=1e-2):
   184:         y = y.astype(np.float32).reshape(-1)
   185:         m = {k: np.zeros_like(v) for k, v in self._params().items()}
   186:         v = {k: np.zeros_like(p) for k, p in self._params().items()}
   187:         b1_, b2_, eps, t = 0.9, 0.999, 1e-8, 0
   188:         for _ in range(epochs):
   189:             t += 1
   190:             pred = self.forward(X)
   191:             err = (pred - y) / max(1, len(X))
   192:             dW2 = self._a1.T @ err.reshape(-1, 1)
   193:             db2 = err.sum(keepdims=True)
   194:             dA1 = err.reshape(-1, 1) @ self.W2.T
   195:             dZ1 = dA1 * (self._z1 > 0)
   196:             dW1 = X.T @ dZ1
   197:             db1 = dZ1.sum(axis=0)
   198:             grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
   199:             for k, g in grads.items():
   200:                 m[k] = b1_ * m[k] + (1 - b1_) * g
   201:                 v[k] = b2_ * v[k] + (1 - b2_) * (g * g)
   202:                 mhat = m[k] / (1 - b1_ ** t)
   203:                 vhat = v[k] / (1 - b2_ ** t)
   204:                 setattr(self, k, getattr(self, k) - lr * mhat / (np.sqrt(vhat) + eps))
   205: 
   206:     def _params(self):
   207:         return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}
   208: 
   209: 
   210: class NASOptimizer:
   211:     """BANANAS — predictor-guided sample-efficient NAS.
   212: 
   213:     Strategy:
   214:     1. Warm start with N0=10 random architectures.
   215:     2. Fit an ensemble of M=5 small MLPs on path-encoded (arch, val_acc) pairs.
   216:     3. Each remaining step: draw a large random pool of candidates, score
   217:        them with ensemble-mean predictions, pick the top unseen candidate,
   218:        query its val accuracy, refit the ensemble.
   219:     """
   220: 
   221:     def __init__(self, api, num_epochs, seed):
   222:         self.api = api
   223:         self.num_epochs = num_epochs
   224:         self.seed = seed
   225: 
   226:         self.warm_start = min(10, num_epochs)
   227:         self.ensemble_size = 5
   228:         self.candidate_pool = 500
   229: 
   230:         self.seen = {}           # arch_tuple -> val_acc
   231:         self.best_arch = None
   232:         self.best_val_acc = -1.0
   233: 
   234:     def _record(self, arch, val_acc):
   235:         self.seen[tuple(arch)] = val_acc
   236:         if val_acc > self.best_val_acc:
   237:             self.best_val_acc = val_acc
   238:             self.best_arch = list(arch)
   239: 
   240:     def _fit_ensemble(self):
   241:         X = np.stack([path_encoding(list(a)) for a in self.seen])
   242:         y = np.array([self.seen[a] for a in self.seen], dtype=np.float32)
   243:         ensemble = []
   244:         for i in range(self.ensemble_size):
   245:             mlp = _TinyMLP(X.shape[1], hidden=64, seed=self.seed + i + 1)
   246:             mlp.fit(X, y, epochs=200, lr=1e-2)
   247:             ensemble.append(mlp)
   248:         return ensemble
   249: 
   250:     def _propose_next(self):
   251:         ensemble = self._fit_ensemble()
   252:         # Large random candidate pool
   253:         cands = []
   254:         while len(cands) < self.candidate_pool:
   255:             a = random_architecture()
   256:             t = tuple(a)
   257:             if t not in self.seen:
   258:                 cands.append(a)
   259:         Xc = np.stack([path_encoding(a) for a in cands])
   260:         preds = np.mean([m.forward(Xc) for m in ensemble], axis=0)
   261:         idx = int(np.argmax(preds))
   262:         return cands[idx]
   263: 
   264:     def search_step(self, epoch):
   265:         if epoch < self.warm_start or len(self.seen) < 2:
   266:             arch = random_architecture()
   267:             while tuple(arch) in self.seen:
   268:                 arch = random_architecture()
   269:         else:
   270:             arch = self._propose_next()
   271: 
   272:         val_acc = self.api.query_val_accuracy(arch)
   273:         self._record(arch, val_acc)
   274: 
   275:         return {
   276:             "best_val_acc": self.best_val_acc,
   277:             "queries": self.api.query_count,
   278:             "current_val_acc": val_acc,
   279:         }
   280: 
   281:     def get_best_architecture(self):
   282:         return self.best_arch
   283: 
   284: # =====================================================================
   285: # FIXED: Main entry point — search + evaluation
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
