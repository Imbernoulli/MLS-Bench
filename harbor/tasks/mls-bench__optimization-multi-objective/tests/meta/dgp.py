"""Held-out problem/scoring module for optimization-multi-objective (NOT agent-visible).

This module lives OUTSIDE every path that is bind-mounted into the agent's
container (workspace package dir, task dir, data binds). It is imported only by
the host-side ``parser.py`` (native) / Harbor verifier and, host-side, by
``edits/mid_edit.py`` to pre-generate the per-run problem specs. It is never
imported by the agent-editable ``deap/custom_moea.py``.

It holds the benchmark problem identities (ZDT/DTLZ), their analytic true Pareto
fronts (``generate_pareto_front``), and the IGD / hypervolume / spread metrics —
so the agent's editable strategy can neither name the eval problem nor recover
the analytic front and emit it as its "discovered" population.

The objective-function spec the MOEA must optimize IS given to the runner (it is
legitimate — evaluating candidates is the task), but it is handed over as an
opaque marshalled black-box ``f(individual) -> objectives`` together with the
numeric problem configuration, carrying no problem name. The analytic front and
the metrics stay here, host-side.

The ``PROBLEMS`` config, ``generate_pareto_front`` and the metric bodies are
byte-identical to the originals that used to live in
``edits/custom_template.py`` — so every honest result is reproduced exactly.
"""

import base64
import marshal

import numpy as np

from deap import benchmarks
from deap.benchmarks import tools as btools


# ================================================================
# Problem definitions (ZDT / DTLZ) — byte-identical to the originals.
# ================================================================

PROBLEMS = {
    "zdt1": {
        "func": benchmarks.zdt1,
        "n_var": 30,
        "n_obj": 2,
        "bounds": (0.0, 1.0),
        "pop_size": 100,
        "n_gen": 200,
        "ref_point": [1.1, 1.1],
        "description": "ZDT1: convex Pareto front, 30 variables, 2 objectives",
    },
    "zdt3": {
        "func": benchmarks.zdt3,
        "n_var": 30,
        "n_obj": 2,
        "bounds": (0.0, 1.0),
        "pop_size": 100,
        "n_gen": 200,
        "ref_point": [1.1, 1.1],
        "description": "ZDT3: disconnected Pareto front, 30 variables, 2 objectives",
    },
    "dtlz2": {
        "func": lambda ind: benchmarks.dtlz2(ind, 3),
        "n_var": 12,
        "n_obj": 3,
        "bounds": (0.0, 1.0),
        "pop_size": 120,
        "n_gen": 250,
        "ref_point": [1.5, 1.5, 1.5],
        "description": "DTLZ2: spherical Pareto front, 12 variables, 3 objectives",
    },
    "dtlz1": {
        "func": lambda ind: benchmarks.dtlz1(ind, 3),
        "n_var": 7,
        "n_obj": 3,
        "bounds": (0.0, 1.0),
        "pop_size": 120,
        "n_gen": 400,
        "ref_point": [1.0, 1.0, 1.0],
        "description": "DTLZ1: linear Pareto front with many local fronts, 7 variables, 3 objectives",
    },
}

# Stable opaque alias <-> problem-name mapping. The runner only ever sees the
# opaque key (``p0``..``p3``) via the ``ENV`` env var and the pre-generated spec
# file; the problem name never enters the agent container.
ALIASES = {
    "zdt1": "p0",
    "zdt3": "p1",
    "dtlz2": "p2",
    "dtlz1": "p3",
}
ALIAS_TO_PROBLEM = {v: k for k, v in ALIASES.items()}


def generate_pareto_front(problem_name: str, n_points: int = 500) -> np.ndarray:
    """Generate reference Pareto front points for IGD computation."""
    if problem_name == "zdt1":
        x = np.linspace(0, 1, n_points)
        return np.column_stack([x, 1 - np.sqrt(x)])
    elif problem_name == "zdt3":
        # ZDT3 has a disconnected front
        regions = [
            (0.0, 0.0830),
            (0.1822, 0.2577),
            (0.4093, 0.4538),
            (0.6183, 0.6525),
            (0.8233, 0.8518),
        ]
        points = []
        per_region = n_points // len(regions)
        for lo, hi in regions:
            x = np.linspace(lo, hi, per_region)
            f1 = x
            f2 = 1 - np.sqrt(x) - x * np.sin(10 * np.pi * x)
            points.append(np.column_stack([f1, f2]))
        return np.vstack(points)
    elif problem_name == "dtlz2":
        # Uniform points on first octant of unit sphere
        points = []
        ns = int(np.sqrt(n_points)) + 1
        for i in range(ns):
            for j in range(ns):
                theta1 = (i / max(ns - 1, 1)) * np.pi / 2
                theta2 = (j / max(ns - 1, 1)) * np.pi / 2
                f1 = np.cos(theta1) * np.cos(theta2)
                f2 = np.cos(theta1) * np.sin(theta2)
                f3 = np.sin(theta1)
                points.append([f1, f2, f3])
        return np.array(points[:n_points])
    elif problem_name == "dtlz1":
        # Pareto front lies on the plane sum(f_i) = 0.5
        points = []
        ns = int(np.sqrt(n_points)) + 1
        for i in range(ns):
            for j in range(ns - i):
                f1 = i / max(ns - 1, 1) * 0.5
                f2 = j / max(ns - 1, 1) * 0.5
                f3 = 0.5 - f1 - f2
                if f3 >= -1e-8:
                    points.append([f1, f2, max(f3, 0.0)])
        return np.array(points[:n_points])
    else:
        raise ValueError(f"Unknown problem: {problem_name}")


# ================================================================
# Metrics — byte-identical to the originals.
# ================================================================

def _hv_2d(points, ref):
    """Exact 2D hypervolume via non-dominated sweep."""
    pts = points[(points[:, 0] < ref[0]) & (points[:, 1] < ref[1])]
    if len(pts) == 0:
        return 0.0
    pts = pts[pts[:, 0].argsort()]
    nd = [pts[0]]
    for p in pts[1:]:
        if p[1] < nd[-1][1]:
            nd.append(p)
    nd = np.array(nd)
    hv = 0.0
    prev_y = ref[1]
    for p in nd:
        width = ref[0] - p[0]
        hv += width * (prev_y - p[1])
        prev_y = p[1]
    return hv


def _hv_3d(points, ref):
    """Exact 3D hypervolume via z-slicing + 2D sweep."""
    mask = np.all(points < ref, axis=1)
    pts = points[mask]
    if len(pts) == 0:
        return 0.0
    # Sort by z ascending: as z increases, more points become active
    order = np.argsort(pts[:, 2])
    pts = pts[order]
    hv = 0.0
    active_2d = []
    for i in range(len(pts)):
        active_2d.append(pts[i, :2])
        z_lo = pts[i, 2]
        z_hi = pts[i + 1, 2] if i + 1 < len(pts) else ref[2]
        dz = z_hi - z_lo
        if dz > 0:
            hv += _hv_2d(np.array(active_2d), ref[:2]) * dz
    return hv


def compute_hypervolume(front_values, ref_point):
    """Robust hypervolume computation that works for 2D and 3D.

    Host-side variant operating directly on the array of objective values for
    the final non-dominated front (instead of DEAP individuals). The numerical
    body is byte-identical to the original ``compute_hypervolume``.
    """
    front_values = np.asarray(front_values, dtype=np.float64)
    ref = np.array(ref_point, dtype=np.float64)
    # Filter out points not dominated by ref
    mask = np.all(front_values < ref, axis=1)
    front_values = front_values[mask]
    if len(front_values) == 0:
        return 0.0
    n_obj = front_values.shape[1]
    if n_obj == 2:
        return _hv_2d(front_values, ref)
    elif n_obj == 3:
        return _hv_3d(front_values, ref)
    return 0.0


def compute_spread(front_values: np.ndarray) -> float:
    """Compute spread (Delta) metric for a 2D front.

    Measures the extent and uniformity of the Pareto front approximation.
    Lower is better. For >2 objectives, returns average pairwise distance std.
    """
    if len(front_values) < 2:
        return float("inf")

    n_obj = front_values.shape[1]
    if n_obj == 2:
        # Sort by first objective
        sorted_idx = np.argsort(front_values[:, 0])
        sorted_front = front_values[sorted_idx]
        # Consecutive distances
        dists = np.sqrt(np.sum(np.diff(sorted_front, axis=0) ** 2, axis=1))
        if len(dists) == 0:
            return float("inf")
        d_mean = np.mean(dists)
        if d_mean < 1e-12:
            return float("inf")
        spread = np.sum(np.abs(dists - d_mean)) / (len(dists) * d_mean)
        return float(spread)
    else:
        # For many-objective: use spacing metric
        from scipy.spatial.distance import cdist

        dist_matrix = cdist(front_values, front_values)
        np.fill_diagonal(dist_matrix, np.inf)
        min_dists = np.min(dist_matrix, axis=1)
        d_mean = np.mean(min_dists)
        if d_mean < 1e-12:
            return float("inf")
        spread = np.sqrt(np.mean((min_dists - d_mean) ** 2)) / d_mean
        return float(spread)


def compute_igd(front_values: np.ndarray, problem_name: str) -> float:
    """IGD of a front-value array against the analytic true Pareto front.

    Uses the same ``deap.benchmarks.tools.igd`` call and the same 500-point
    reference front as the original pipeline.
    """
    pf_ref = generate_pareto_front(problem_name, n_points=500)
    try:
        return float(btools.igd(np.asarray(front_values), pf_ref))
    except Exception:
        return float("inf")


# ================================================================
# Black-box objective evaluator (handed to the in-container runner).
# ================================================================
#
# The runner must be able to EVALUATE candidate solutions (that is the task),
# but it must not be able to recover the analytic Pareto front nor the problem
# name. We hand it a marshalled, name-free code object that dispatches into
# ``deap.benchmarks`` by an opaque integer ``kind`` carried in the spec. The
# code object is reconstructed and called purely as a black box ``f(ind)``.

# kind ordering must stay in sync with _KIND_ORDER below.
_KIND_ORDER = ["zdt1", "zdt3", "dtlz2", "dtlz1"]


def _objective_kernel(individual, kind, n_obj):
    """Pure black-box objective dispatch (marshalled into the spec).

    Reconstructed and called in-container with no problem name in scope. Returns
    a tuple of objective values (all minimized).
    """
    import deap.benchmarks as _b
    if kind == 0:
        return tuple(_b.zdt1(individual))
    if kind == 1:
        return tuple(_b.zdt3(individual))
    if kind == 2:
        return tuple(_b.dtlz2(individual, n_obj))
    if kind == 3:
        return tuple(_b.dtlz1(individual, n_obj))
    raise ValueError("unknown objective kind")


def _evaluator_blob() -> str:
    """Base64 of the marshalled black-box objective code object."""
    return base64.b64encode(marshal.dumps(_objective_kernel.__code__)).decode("ascii")


# ================================================================
# Helpers used by the input pre-generator and the host-side scorer.
# ================================================================

def gen_problem(problem: str, seed: int = 42) -> dict:
    """Return the agent-visible problem SPEC the MOEA must optimize.

    The spec carries the numeric problem configuration plus a marshalled
    black-box objective evaluator. It deliberately contains NO problem name and
    NO Pareto front, so the pre-generator that writes the workspace spec never
    persists the answer.
    """
    cfg = PROBLEMS[problem]
    return {
        "alias": ALIASES[problem],
        "seed": int(seed),
        "n_var": int(cfg["n_var"]),
        "n_obj": int(cfg["n_obj"]),
        "bounds": [float(cfg["bounds"][0]), float(cfg["bounds"][1])],
        "pop_size": int(cfg["pop_size"]),
        "n_gen": int(cfg["n_gen"]),
        "kind": _KIND_ORDER.index(problem),
        "evaluator": _evaluator_blob(),
    }


def truth(problem: str, seed: int = 42) -> dict:
    """Held-out ground truth for the host-side scorer: front + ref_point."""
    cfg = PROBLEMS[problem]
    return {
        "ref_point": [float(v) for v in cfg["ref_point"]],
        "n_obj": int(cfg["n_obj"]),
    }


def score_front(front_values, problem: str, seed: int = 42) -> dict:
    """Compute (hv, igd, spread) for a final non-dominated front-value array.

    Uses the held-out analytic Pareto front and ref_point. Numerically
    byte-identical to the original in-file metric computation.
    """
    cfg = PROBLEMS[problem]
    ref_point = cfg["ref_point"]
    front_values = np.asarray(front_values, dtype=np.float64)
    hv = float(compute_hypervolume(front_values, ref_point))
    igd = compute_igd(front_values, problem)
    spread = compute_spread(front_values)
    return {"hv": hv, "igd": igd, "spread": spread}
