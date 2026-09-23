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
import json
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
    """Client for NAS-Bench-201 validation queries with a hard budget.

    The validation table is NOT in this process: a separate FIXED oracle
    process (the harness's nas_oracle_entry.py) holds it, counts every query
    and enforces the budget; this object only forwards architecture strings
    to it over a pipe. ``query_count`` here is a local mirror for logging;
    the oracle's own count is authoritative. The held-out TEST accuracy of
    the final architecture is looked up by the harness afterwards, outside
    this process (see the FINAL_ARCH report in the main entry point).
    """

    def __init__(self, oracle, dataset_key, query_budget):
        self.dataset_key = dataset_key
        self.query_budget = int(query_budget)
        self.query_count = 0
        self._oracle = oracle  # _OracleClient (FIXED, defined below)

    @property
    def remaining_budget(self):
        return max(0, self.query_budget - self.query_count)

    def query_val_accuracy(self, op_indices):
        """Query validation accuracy (counts against the budget).

        For cifar10, validation accuracy is from the 'cifar10-valid' split.
        For cifar100 and ImageNet16-120, validation accuracy uses 'eval_acc1es'
        from the respective split (standard NAS-Bench-201 search protocol).
        The oracle process counts this query and refuses every query past
        the budget, whatever the local mirror ``query_count`` says (it may
        be reset or overwritten; that does not buy extra queries).
        """
        if self.query_count >= self.query_budget:
            raise BudgetExceededError(
                f"Validation query budget of {self.query_budget} exhausted."
            )
        self.query_count += 1
        return self._oracle.query(op_indices_to_arch_str(op_indices))


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
# FIXED: Main entry point — search + final-architecture report
# =====================================================================
class _OracleClient:
    """FIXED pipe client for the out-of-process validation oracle.

    The harness (nas_oracle_entry.py) starts this program with two pipe file
    descriptors in MLSBENCH_NAS_ORACLE_FDS ("<request>,<reply>"). The oracle
    process on the other end holds the only copy of the validation table,
    counts every request and refuses those past the budget.
    """

    def __init__(self, spec):
        req_fd, resp_fd = (int(x) for x in spec.split(","))
        self._req = os.fdopen(req_fd, "w", encoding="utf-8", newline="\n")
        self._resp = os.fdopen(resp_fd, "r", encoding="utf-8", newline="\n")

    def query(self, arch_str):
        self._req.write(arch_str + "\n")
        self._req.flush()
        reply = self._resp.readline().rstrip("\n")
        if reply.startswith("OK "):
            return float(reply[3:])
        if reply.startswith("BUDGET"):
            raise BudgetExceededError(
                f"Validation query budget of {reply.split()[-1]} exhausted "
                "(refused by the oracle)."
            )
        if reply == "KEYERR":
            raise KeyError(arch_str)
        raise RuntimeError(f"validation oracle failed (reply={reply!r})")


def _connect_oracle():
    """Connect to the harness's out-of-process validation oracle."""
    spec = os.environ.get("MLSBENCH_NAS_ORACLE_FDS")
    if not spec:
        print("ERROR: no validation oracle (MLSBENCH_NAS_ORACLE_FDS unset); "
              "run this program through the harness's nas_oracle_entry.py.",
              flush=True)
        sys.exit(1)
    return _OracleClient(spec)


def _main():
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

    # ── Connect to the out-of-process validation oracle (the table and the ──
    # ── authoritative query count live there; test accuracies stay outside) ──
    oracle = _connect_oracle()

    # ── Create benchmark API with strict budget ──
    api = BenchmarkAPI(oracle, dataset_key, query_budget=num_epochs)

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

    # ── Final-architecture report (the held-out test accuracy is looked ──
    # ── up by the harness outside this process)                         ──
    best_arch = optimizer.get_best_architecture()
    if best_arch is None:
        print("ERROR: No architecture found during search!", flush=True)
        sys.exit(1)
    if not is_valid_arch(best_arch):
        print(f"ERROR: Returned architecture {best_arch} is invalid.", flush=True)
        sys.exit(1)

    best_arch_str = op_indices_to_arch_str(best_arch)
    total_queries = api.query_count
    total_time = time.time() - start_time

    print(f"\n{'='*60}", flush=True)
    print(f"Search complete on {env_name} (dataset={dataset_key})", flush=True)
    print(f"Best architecture: {best_arch} -> {best_arch_str}", flush=True)
    print(f"Total val queries used: {total_queries} / {num_epochs}", flush=True)
    print(f"Total time: {total_time:.1f}s", flush=True)
    print(f"{'='*60}", flush=True)

    # Report the chosen architecture for external (held-out) evaluation
    print(f"FINAL_ARCH arch={best_arch_str} queries={total_queries}", flush=True)


if __name__ == "__main__":
    _main()
