# MLS-Bench: optimization-nas

# Sample-Efficient Neural Architecture Search

## Objective
Design and implement a novel **sample-efficient** NAS optimizer that discovers high-performing architectures under a **strict query budget**. Your code goes in the `NASOptimizer` class in `custom_nas_search.py`.

## Research Question
Under a strict query budget per search, how can a search strategy maximize the expected quality of the best-found architecture?

This is the regime in which real-world NAS is actually hard: the agent can only query a small number of architectures, so naïve enumeration is impossible and algorithmic differences are load-bearing.

## Search Space
- Cell-based search space: 4 nodes, 6 edges, 5 operations per edge.
- Operations: `skip_connect, none, nor_conv_3x3, nor_conv_1x1, avg_pool_3x3`.
- 5^6 = 15,625 architectures total.
- An architecture is represented as a list of 6 integers in `[0, 4]`.

## Protocol
- A small validation-query budget is enforced per run (`NAS_EPOCHS`, default 30). The harness enforces this; exceeding it aborts the run.
- The agent should maintain `self.best_arch` so that `get_best_architecture()` returns the chosen architecture at the end.

## What Counts as a Contribution
Acceptable research directions (this list is not exhaustive):
- **Better acquisition functions**: e.g. UCB / EI over a learned predictor, Thompson sampling, information-theoretic criteria.
- **Better surrogate models**: predictors over path-encoded architectures, MLP ensembles, zero-cost proxy hybrids.
- **Smarter exploration–exploitation mixing**: local search, portfolio methods, warm-started evolution.
- **Encoding choices**: adjacency vs path encoding.

What does **not** count:
- Increasing the effective budget (e.g. re-querying the same architecture, wrapping queries, etc.). The harness counts every call to `api.query_val_accuracy` and will terminate after the budget runs out.
- Hard-coding known good architectures.

## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/naslib/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be rejected.

- `naslib/custom_nas_search.py`
- editable lines **163–234**

## Readable Context

### `naslib/custom_nas_search.py`  [EDITABLE — lines 163–234 only]

```python
   160: # =====================================================================
   161: # EDITABLE: NAS Optimizer — implement your search strategy here
   162: # =====================================================================
   163: class NASOptimizer:
   164:     """Sample-efficient NAS search strategy.
   165:
   166:     Implement a search algorithm that maximizes the test quality of the
   167:     best-found architecture under a STRICT validation-query budget
   168:     (self.num_epochs).
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
   189:     harness to finally evaluate.
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
```

## Tips

- Keep the function/class signatures of the editable regions identical;
  the editable region is imported by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.

Good luck.
