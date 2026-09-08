"""Held-out scoring module for ml-symbolic-regression (NOT agent-visible).

Lives outside every path bind-mounted into the agent container; imported only
by the host-side parser.py (native) / Harbor verifier and by the input
pre-generator in ``edits/mid_edit.py`` (which runs host-side at setup time).
It holds the benchmark TARGET FUNCTIONS (the closed-form expressions the GP is
supposed to rediscover), the data generator, and the R2 metric — so the
editable GP operators can neither name the eval benchmarks nor read off the
target formula to build the exact expression tree.

The ``BENCHMARKS`` dict, ``generate_data`` body, and ``r2_score`` formula are
byte-identical to the originals that used to live in
``edits/custom_template.py`` — so the data, and therefore every honest result,
is reproduced exactly. The agent's GP fits only to the (X_train, y_train)
SAMPLES that ``gen_input`` returns; it never receives the closed form or the
benchmark name.
"""

import hashlib

import numpy as np


# Host-only salt: makes the opaque per-benchmark input-file token impossible to
# reverse-map back to the famous benchmark name from inside the agent container
# (the salt never leaves this host-only module).
_TASK_ID_SALT = "mlsbench-sr-v2"


def task_id(benchmark):
    """Stable, name-free opaque token used to key the pre-generated inputs.

    Used by the host-side input pre-generator (mid_edit) and baked into the
    per-benchmark run scripts. Carries no information about which target
    function is in use.
    """
    return hashlib.sha1((_TASK_ID_SALT + benchmark).encode()).hexdigest()[:12]


# ============================================================
# Benchmark Data (target functions live here, host-only)
# ============================================================

BENCHMARKS = {
    'nguyen7': {
        'func': lambda X: np.log(X[:, 0] + 1) + np.log(X[:, 0] ** 2 + 1),
        'n_features': 1,
        'train_range': (0.0, 2.0),
        'n_train': 20,
        'test_range': (-0.5, 2.5),
        'n_test': 100,
    },
    'nguyen10': {
        'func': lambda X: 2 * np.sin(X[:, 0]) * np.cos(X[:, 1]),
        'n_features': 2,
        'train_range': (0.0, 2 * np.pi),
        'n_train': 100,
        'test_range': (0.0, 2 * np.pi),
        'n_test': 400,
    },
    'koza3': {
        'func': lambda X: X[:, 0] ** 5 - 2 * X[:, 0] ** 3 + X[:, 0],
        'n_features': 1,
        'train_range': (-1.0, 1.0),
        'n_train': 20,
        'test_range': (-1.0, 1.0),
        'n_test': 100,
    },
}


def generate_data(benchmark_name, seed=42):
    """Generate train/test data for a benchmark function."""
    bench = BENCHMARKS[benchmark_name]
    n_features = bench['n_features']
    lo, hi = bench['train_range']

    if n_features == 1:
        X_train = np.linspace(lo, hi, bench['n_train']).reshape(-1, 1)
        lo_t, hi_t = bench['test_range']
        X_test = np.linspace(lo_t, hi_t, bench['n_test']).reshape(-1, 1)
    else:
        n_per_dim = int(round(bench['n_train'] ** (1.0 / n_features)))
        grids = [np.linspace(lo, hi, n_per_dim) for _ in range(n_features)]
        X_train = np.array(np.meshgrid(*grids)).T.reshape(-1, n_features)
        lo_t, hi_t = bench['test_range']
        n_per_dim_t = int(round(bench['n_test'] ** (1.0 / n_features)))
        grids_t = [np.linspace(lo_t, hi_t, n_per_dim_t) for _ in range(n_features)]
        X_test = np.array(np.meshgrid(*grids_t)).T.reshape(-1, n_features)

    y_train = bench['func'](X_train)
    y_test = bench['func'](X_test)
    return X_train, y_train, X_test, y_test, n_features


# ============================================================
# Metric (same formula as before)
# ============================================================

def r2_score(y_true, y_pred):
    """Compute R-squared score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-15:
        return 1.0 if ss_res < 1e-15 else 0.0
    return max(1.0 - ss_res / ss_tot, 0.0)  # SRBench floor: clip blowups


# ============================================================
# Helpers for the input pre-generator and the host-side scorer
# ============================================================

def gen_input(benchmark, seed=42):
    """Return the agent-visible inputs (X_train, y_train, X_test).

    The GP legitimately needs y_train SAMPLES to drive its fitness, so these
    are provided. The closed-form target ``func`` and the test labels are NOT
    returned, so the pre-generator that writes the agent's input files never
    persists the answer.
    """
    X_train, y_train, X_test, _y_test, _n = generate_data(benchmark, seed)
    return X_train, y_train, X_test


def truth(benchmark, seed=42):
    """Return the held-out ground truth (y_test) for the host-side scorer."""
    _Xtr, _ytr, _Xte, y_test, _n = generate_data(benchmark, seed)
    return y_test
