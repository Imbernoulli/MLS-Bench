# Custom NAS optimizer for MLS-Bench (NAS-Bench-201, sample-efficient regime)
#
# EDITABLE section: NASOptimizer class — implement your search strategy.
# FIXED sections: everything else (search space, benchmark API, evaluation loop).
#
# The NAS-Bench-201 search space has 15625 architectures (5 ops, 6 edges).
# Evaluation is tabular — query the benchmark for any architecture's accuracy.
# No actual neural network training is needed.
#
# IMPORTANT: You have a STRICT budget of NAS_EPOCHS validation queries
# (default 30). The BenchmarkAPI enforces this and will raise
# BudgetExceededError if you exceed it. One final test query at the end is
# free and not counted against the budget.
import os
import sys
import time
import random
import pickle
import copy
import numpy as np
from pathlib import Path


# =====================================================================
# FIXED: NAS-Bench-201 Search Space Definition
# =====================================================================
NUM_EDGES = 6
NUM_OPS = 5
OP_NAMES = ["skip_connect", "none", "nor_conv_3x3", "nor_conv_1x1", "avg_pool_3x3"]

# Edge list: (source, target) for the 4-node cell
# Node 0: input, Nodes 1-2: intermediate, Node 3: output
EDGE_LIST = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))

# Dataset name mapping for the benchmark lookup
DATASET_MAP = {
    "cifar10": "cifar10",
    "cifar100": "cifar100",
    "imagenet16": "ImageNet16-120",
}


class BudgetExceededError(RuntimeError):
    """Raised when the validation query budget is exhausted."""


def op_indices_to_arch_str(op_indices):
    """Convert a list of 6 op indices to the NAS-Bench-201 architecture string."""
    edge_op_dict = {
        edge: OP_NAMES[op] for edge, op in zip(EDGE_LIST, op_indices)
    }
    op_edge_list = [
        "{}~{}".format(edge_op_dict[(i, j)], i - 1)
        for i, j in sorted(edge_op_dict, key=lambda x: x[1])
    ]
    return "|{}|+|{}|{}|+|{}|{}|{}|".format(*op_edge_list)


def is_valid_arch(op_indices):
    """Check architecture validity (not all-zero on any path)."""
    # none=1 in OP_NAMES; reject if all edges to node 3 are 'none'
    # or all edges from node 1 are 'none'
    return not ((op_indices[0] == op_indices[1] == op_indices[2] == 1) or
                (op_indices[2] == op_indices[4] == op_indices[5] == 1))


def random_architecture():
    """Sample a random valid architecture as a list of 6 op indices."""
    while True:
        op_indices = [random.randint(0, NUM_OPS - 1) for _ in range(NUM_EDGES)]
        if is_valid_arch(op_indices):
            return op_indices


def mutate_architecture(parent_op_indices):
    """Mutate one random edge of the parent architecture."""
    op_indices = list(parent_op_indices)
    edge = random.randint(0, NUM_EDGES - 1)
    available = [o for o in range(NUM_OPS) if o != parent_op_indices[edge]]
    op_indices[edge] = random.choice(available)
    return op_indices


def get_neighbors(op_indices):
    """Get all 1-edit-distance neighbors of an architecture."""
    neighbors = []
    for edge in range(NUM_EDGES):
        for op in range(NUM_OPS):
            if op != op_indices[edge]:
                nbr = list(op_indices)
                nbr[edge] = op
                neighbors.append(nbr)
    return neighbors


def path_encoding(op_indices):
    """Path encoding of a NAS-Bench-201 cell (White et al., 2020).

    Enumerates every op-labeled path from input to output and returns a binary
    indicator vector of length NUM_OPS**3 + NUM_OPS**2 + NUM_OPS (paths of
    length 1, 2, 3 respectively). Useful as input to predictor models.
    """
    # Edges: 0:(1,2) 1:(1,3) 2:(1,4) 3:(2,3) 4:(2,4) 5:(3,4)
    # i.e. from input(node 1) to output(node 4)
    o = op_indices
    enc_len = NUM_OPS ** 3 + NUM_OPS ** 2 + NUM_OPS
    v = np.zeros(enc_len, dtype=np.float32)
    # length-1 paths (direct 1->4)
    v[o[2]] = 1.0
    # length-2 paths (1->2->4, 1->3->4)
    offset = NUM_OPS
    v[offset + o[0] * NUM_OPS + o[4]] = 1.0
    v[offset + o[1] * NUM_OPS + o[5]] = 1.0
    # length-3 paths (1->2->3->4)
    offset = NUM_OPS + NUM_OPS ** 2
    v[offset + o[0] * NUM_OPS ** 2 + o[3] * NUM_OPS + o[5]] = 1.0
    return v


class BenchmarkAPI:
    """Wrapper for querying NAS-Bench-201 with a hard validation-query budget."""

    def __init__(self, data, dataset_key, query_budget):
        self.data = data
        self.dataset_key = dataset_key
        self.query_budget = int(query_budget)
        self.query_count = 0
        self._cache = {}  # repeated queries don't cost extra but still count

    @property
    def remaining_budget(self):
        return max(0, self.query_budget - self.query_count)

    def query_val_accuracy(self, op_indices):
        """Query validation accuracy (counts against the budget).

        For cifar10, validation accuracy is from the 'cifar10-valid' split.
        For cifar100 and ImageNet16-120, validation accuracy uses 'eval_acc1es'
        from the respective split (standard NAS-Bench-201 search protocol).
        """
        if self.query_count >= self.query_budget:
            raise BudgetExceededError(
                f"Validation query budget of {self.query_budget} exhausted."
            )
        self.query_count += 1
        arch_str = op_indices_to_arch_str(op_indices)
        if self.dataset_key == "cifar10":
            return self.data[arch_str]["cifar10-valid"]["eval_acc1es"]
        else:
            return self.data[arch_str][self.dataset_key]["eval_acc1es"]

    # --- Harness-only methods (not counted against the agent's budget) ---

    def _query_test_accuracy_unbudgeted(self, op_indices):
        """Query final test accuracy — only called by the harness after search."""
        arch_str = op_indices_to_arch_str(op_indices)
        return self.data[arch_str][self.dataset_key]["eval_acc1es"]


# =====================================================================
# EDITABLE: NAS Optimizer — implement your search strategy here
# =====================================================================
class NASOptimizer:
    """Sample-efficient NAS search strategy.

    Implement a search algorithm that maximizes the test accuracy of the
    best-found architecture under a STRICT validation-query budget
    (self.num_epochs, default 30).

    The search space has 15625 architectures (5 ops x 6 edges). Each
    architecture is a list of 6 integers in [0, 4].

    Available helper functions (defined above, fixed):
        random_architecture()                  -> list[int]  (random valid arch)
        mutate_architecture(parent)            -> list[int]  (1-edge mutation)
        get_neighbors(op_indices)              -> list[list[int]]  (all 1-edit neighbors)
        is_valid_arch(op_indices)              -> bool
        op_indices_to_arch_str(op_indices)     -> str
        path_encoding(op_indices)              -> np.ndarray (features for predictors)

    The benchmark API (self.api) provides ONE budgeted method:
        api.query_val_accuracy(op_indices)     -> float   (costs 1 query)
        api.query_count                        -> int     (queries used so far)
        api.remaining_budget                   -> int     (queries left)

    The harness will call search_step(epoch) up to self.num_epochs times.
    After each step, you should maintain self.best_arch so that
    get_best_architecture() returns the architecture you most want the
    harness to finally test (on the unbudgeted test split).
    """

    def __init__(self, api, num_epochs, seed):
        """Initialize the optimizer.

        Args:
            api: BenchmarkAPI (with budget = num_epochs validation queries).
            num_epochs: Total number of allowed validation queries (budget).
            seed: Random seed for reproducibility.
        """
        self.api = api
        self.num_epochs = num_epochs
        self.seed = seed

        # TODO: Initialize your search state here
        self.best_arch = None
        self.best_val_acc = -1.0

    def search_step(self, epoch):
        """Run one step of the search algorithm.

        Args:
            epoch: Current search iteration (0-indexed)

        Returns:
            dict: Metrics to log, must include 'best_val_acc' and 'queries'.
        """
        # Placeholder: random search (replace with your algorithm)
        arch = random_architecture()
        val_acc = self.api.query_val_accuracy(arch)

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_arch = arch

        return {
            "best_val_acc": self.best_val_acc,
            "queries": self.api.query_count,
            "current_val_acc": val_acc,
        }

    def get_best_architecture(self):
        """Return the architecture the harness will test (unbudgeted)."""
        return self.best_arch


# =====================================================================
# FIXED: Main entry point — search + evaluation
# =====================================================================
if __name__ == "__main__":
    # ── Configuration from environment ──
    seed = int(os.environ.get("SEED", 42))
    output_dir = os.environ.get("OUTPUT_DIR", "/tmp/nas_output")
    env_name = os.environ.get("ENV", "cifar10")
    num_epochs = int(os.environ.get("NAS_EPOCHS", 30))  # sample-efficient: K=30

    os.makedirs(output_dir, exist_ok=True)

    # ── Seeding ──
    random.seed(seed)
    np.random.seed(seed)

    # ── Map environment name to dataset key ──
    dataset_key = DATASET_MAP.get(env_name)
    if dataset_key is None:
        print(f"ERROR: Unknown environment '{env_name}'. Must be one of: {list(DATASET_MAP.keys())}")
        sys.exit(1)

    # ── Load NAS-Bench-201 benchmark data ──
    _candidates = [
        Path("/workspace/naslib/naslib/data/nb201_all.pickle"),
        Path(__file__).resolve().parent / "naslib" / "data" / "nb201_all.pickle",
        Path(__file__).resolve().parent / "data" / "nb201_all.pickle",
        Path("naslib/data/nb201_all.pickle"),
        Path("data/nb201_all.pickle"),
    ]
    data_path = None
    for _p in _candidates:
        if _p.exists():
            data_path = _p
            break
    if data_path is None:
        print(f"ERROR: Benchmark data not found. Searched: {[str(p) for p in _candidates]}")
        sys.exit(1)

    print(f"Loading NAS-Bench-201 data from {data_path}...", flush=True)
    with open(data_path, "rb") as f:
        nb201_data = pickle.load(f)
    print(f"Loaded {len(nb201_data)} architectures.", flush=True)

    # ── Create benchmark API with strict budget ──
    api = BenchmarkAPI(nb201_data, dataset_key, query_budget=num_epochs)

    # ── Run search ──
    print(f"Starting sample-efficient NAS on {env_name} (dataset={dataset_key}) "
          f"with budget={num_epochs} queries, seed={seed}", flush=True)

    optimizer = NASOptimizer(api, num_epochs, seed)

    start_time = time.time()
    for epoch in range(num_epochs):
        if api.remaining_budget <= 0:
            print(f"Budget exhausted at epoch {epoch}; stopping search.", flush=True)
            break
        try:
            metrics = optimizer.search_step(epoch)
        except BudgetExceededError as e:
            print(f"BUDGET EXCEEDED at epoch {epoch}: {e}", flush=True)
            break

        # Log training metrics every step (K=30 is small)
        elapsed = time.time() - start_time
        metrics_str = " ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        )
        print(f"TRAIN_METRICS epoch={epoch+1} {metrics_str} "
              f"elapsed={elapsed:.1f}s", flush=True)

    # ── Final evaluation (UNBUDGETED) ──
    best_arch = optimizer.get_best_architecture()
    if best_arch is None:
        print("ERROR: No architecture found during search!", flush=True)
        sys.exit(1)
    if not is_valid_arch(best_arch):
        print(f"ERROR: Returned architecture {best_arch} is invalid.", flush=True)
        sys.exit(1)

    best_arch_str = op_indices_to_arch_str(best_arch)
    test_acc = api._query_test_accuracy_unbudgeted(best_arch)
    total_queries = api.query_count
    total_time = time.time() - start_time

    print(f"\n{'='*60}", flush=True)
    print(f"Search complete on {env_name} (dataset={dataset_key})", flush=True)
    print(f"Best architecture: {best_arch} -> {best_arch_str}", flush=True)
    print(f"Total val queries used: {total_queries} / {num_epochs}", flush=True)
    print(f"Total time: {total_time:.1f}s", flush=True)
    print(f"{'='*60}", flush=True)

    # Output test metric for parser
    print(f"TEST_METRICS test_accuracy={test_acc:.4f}", flush=True)
