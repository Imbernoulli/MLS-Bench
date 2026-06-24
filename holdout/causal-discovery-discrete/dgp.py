"""Held-out data + scoring module for causal-discovery-discrete (NOT agent-visible).

This module lives OUTSIDE every path that is bind-mounted into the agent's
container (workspace package dir, task dir, data binds). It is imported only by
the host-side ``parser.py`` (native) and by the Harbor verifier (tests/) — never
by the agent-editable ``custom_algorithm.py`` or by anything inside the agent's
container.

It holds the three secrets that must never reach the agent's editable scope:
  1. the eval network identity (the ``LABEL_SPEC`` map: which bnlearn Bayesian
     network / how many samples each config label corresponds to),
  2. the ground-truth DAG construction (via ``pgmpy.utils.get_example_model``),
  3. the metric (SHD / adjacency P,R / arrow P,R via causallearn).

``_load_and_sample`` and ``_compute_metrics`` are byte-identical to the originals
that used to live in ``edits/data_gen_template.py`` and ``edits/metrics_template.py``
— so the sampled data X, the true DAG, and therefore every honest result are
reproduced exactly.

Dependency note
---------------
The heavy deps (``pgmpy`` to sample / build the truth, ``causallearn`` to score)
are NOT guaranteed to be importable in the harness/CLI Python that runs the
host-side ``parser.py`` and ``mid_edit.py``. They ARE installed in the package
conda env (``mlsbench-causal-bnlearn`` ships pgmpy) and the bundled
``causallearn`` lives under ``vendor/external_packages/causal-bnlearn``. So this
module first tries to do everything in-process; if the heavy imports are not
available it transparently re-runs the exact same code in the package conda env
via ``conda run`` (a self-contained worker), exchanging arrays as base64. The
worker imports the SAME pgmpy / SAME bundled causallearn, so results are
identical regardless of which interpreter the harness happens to use.
"""

import base64
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[2]
_VENDORED_CAUSALLEARN = _PROJECT_ROOT / "vendor" / "external_packages" / "causal-bnlearn"
_PKG_NAME = "causal-bnlearn"
_CONDA_ENV = f"mlsbench-{_PKG_NAME}"

# =====================================================================
# Held-out eval spec (network identity — the secret the agent must not see).
# Keyed by the config test_cmd label. The (network, n_samples) here MUST match
# the original scripts/eval_<net>.sh: cancer/500, child/2000, alarm/5000,
# hailfinder/10000, win95pts/10000.
# =====================================================================
LABEL_SPEC = {
    "Cancer":     ("cancer",     500),
    "Child":      ("child",      2000),
    "Alarm":      ("alarm",      5000),
    "Hailfinder": ("hailfinder", 10000),
    "Win95pts":   ("win95pts",   10000),
}


# =====================================================================
# Core DGP + metric bodies (byte-identical to the originals).
# These are only ever executed where pgmpy + causallearn are importable
# (either the harness Python if it happens to have them, or the conda worker).
# =====================================================================

def _load_and_sample(network_name, n_samples, seed):
    """Load a bnlearn network, sample data, and return (X, true_dag, node_names).

    Byte-identical to the original ``edits/data_gen_template.py:load_and_sample``.
    """
    import numpy as np
    import pandas as pd

    from causallearn.graph.Dag import Dag
    from causallearn.graph.GraphNode import GraphNode

    from pgmpy.utils import get_example_model
    from pgmpy.sampling import BayesianModelSampling

    model = get_example_model(network_name)

    # Sample observational data
    sampler = BayesianModelSampling(model)
    df = sampler.forward_sample(size=n_samples, seed=seed)

    # Sorted node names for consistent ordering
    node_names = sorted(model.nodes())
    n_vars = len(node_names)

    # Integer-encode each variable (deterministic via sorted categories)
    X = np.zeros((n_samples, n_vars), dtype=int)
    for j, name in enumerate(node_names):
        col = df[name]
        categories = sorted(col.unique())
        cat_type = pd.CategoricalDtype(categories=categories, ordered=True)
        X[:, j] = col.astype(cat_type).cat.codes.values

    # Build true DAG as a causal-learn Dag object.
    # Use X1, X2, ... naming to match what causal-learn algorithms produce.
    name_to_idx = {name: i for i, name in enumerate(node_names)}
    dag_nodes = [GraphNode(f"X{i + 1}") for i in range(n_vars)]
    true_dag = Dag(dag_nodes)
    for parent, child in model.edges():
        true_dag.add_directed_edge(dag_nodes[name_to_idx[parent]],
                                   dag_nodes[name_to_idx[child]])

    return X, true_dag, node_names


def _safe_div(numerator, denominator):
    return numerator / denominator if denominator > 0 else 0.0


def _compute_metrics(est_graph, true_dag):
    """Compute SHD and precision/recall metrics on CPDAGs.

    Byte-identical to the original ``edits/metrics_template.py:compute_metrics``
    (the ``_normalize_graph_output`` step is performed by the caller before the
    est graph is serialized).
    """
    from causallearn.graph.AdjacencyConfusion import AdjacencyConfusion
    from causallearn.graph.ArrowConfusion import ArrowConfusion
    from causallearn.utils.DAG2CPDAG import dag2cpdag

    est_cpdag = est_graph
    true_cpdag = dag2cpdag(true_dag)

    from causallearn.graph.SHD import SHD
    shd = SHD(true_cpdag, est_cpdag).get_shd()

    adj = AdjacencyConfusion(true_cpdag, est_cpdag)
    adj_tp = adj.get_adj_tp()
    adj_fp = adj.get_adj_fp()
    adj_fn = adj.get_adj_fn()
    adj_precision = _safe_div(adj_tp, adj_tp + adj_fp)
    adj_recall = _safe_div(adj_tp, adj_tp + adj_fn)

    arrow = ArrowConfusion(true_cpdag, est_cpdag)
    arrow_tp = arrow.get_arrows_tp()
    arrow_fp = arrow.get_arrows_fp()
    arrow_fn = arrow.get_arrows_fn()
    arrow_precision = _safe_div(arrow_tp, arrow_tp + arrow_fp)
    arrow_recall = _safe_div(arrow_tp, arrow_tp + arrow_fn)

    return {
        "shd": int(shd),
        "adj_precision": float(adj_precision),
        "adj_recall": float(adj_recall),
        "arrow_precision": float(arrow_precision),
        "arrow_recall": float(arrow_recall),
    }


# =====================================================================
# In-process worker entrypoints (run where pgmpy+causallearn import OK).
# =====================================================================

def _worker_gen_input(network_name, n_samples, seed):
    """Return ONLY the agent-visible input X (int adjacency-free samples)."""
    X, _true_dag, _names = _load_and_sample(network_name, n_samples, seed)
    return np.ascontiguousarray(X, dtype=np.int64)


def _build_general_graph_from_matrix(n_vars, graph_matrix):
    """Reconstruct a causallearn GeneralGraph from its endpoint ``.graph`` matrix.

    The agent's program serializes ``est.graph`` (the n x n int endpoint matrix
    that AdjacencyConfusion / ArrowConfusion / SHD read directly), together with
    X1..Xn node naming. Assigning ``.graph`` round-trips the estimate exactly, so
    scoring matches what the in-container ``compute_metrics(est_graph, ...)`` used
    to produce.
    """
    from causallearn.graph.GeneralGraph import GeneralGraph
    from causallearn.graph.GraphNode import GraphNode

    nodes = [GraphNode(f"X{i + 1}") for i in range(n_vars)]
    g = GeneralGraph(nodes)
    g.graph = np.asarray(graph_matrix, dtype=int).reshape(n_vars, n_vars)
    return g


def _worker_score(network_name, n_samples, seed, est_matrix):
    """Regenerate the true DAG and score the serialized est endpoint matrix."""
    _X, true_dag, _names = _load_and_sample(network_name, n_samples, seed)
    n_vars = int(np.asarray(est_matrix).shape[0])
    est_graph = _build_general_graph_from_matrix(n_vars, est_matrix)
    return _compute_metrics(est_graph, true_dag)


# =====================================================================
# Heavy-dep availability + transparent conda-worker fallback.
# =====================================================================

def _heavy_imports_available():
    if str(_VENDORED_CAUSALLEARN) not in sys.path:
        sys.path.insert(0, str(_VENDORED_CAUSALLEARN))
    try:
        import pgmpy  # noqa: F401
        import causallearn  # noqa: F401
        from causallearn.utils.DAG2CPDAG import dag2cpdag  # noqa: F401
        return True
    except Exception:
        return False


def _find_conda_exe():
    """Locate the conda executable (mirrors src/mlsbench/cli.py:find_conda_exe)."""
    import shutil

    exe = shutil.which("conda")
    if exe:
        return exe
    hints = [os.environ.get("CONDA_EXE", ""), os.environ.get("MAMBA_EXE", "")]
    prefix = os.environ.get("CONDA_PREFIX", "")
    if prefix:
        hints += [
            str(Path(prefix) / "condabin" / "conda"),
            str(Path(prefix).parent / "condabin" / "conda"),
        ]
    interp = Path(sys.executable).resolve()
    for parent in [interp.parent, *interp.parents]:
        hints.append(str(parent / "condabin" / "conda"))
        hints.append(str(parent / "bin" / "conda"))
    for home in [Path.home()]:
        hints.append(str(home / "miniconda3" / "condabin" / "conda"))
        hints.append(str(home / "anaconda3" / "condabin" / "conda"))
    hints += ["/opt/conda/condabin/conda", "/usr/local/conda/condabin/conda"]
    for c in hints:
        if c and Path(c).exists():
            return c
    return None


# Self-contained worker script run inside the package conda env. It re-imports
# this module from the holdout dir and dispatches one call, reading a JSON
# request on argv and writing a JSON response to stdout (delimited so any
# pgmpy/sampling progress chatter on stderr/stdout is ignored).
_WORKER_SRC = r'''
import sys, os, json, base64, io
import numpy as np
sys.path.insert(0, os.environ["MLSB_HOLDOUT_DIR"])
sys.path.insert(0, os.environ["MLSB_VENDORED_CAUSALLEARN"])
import dgp  # this very module, imported where pgmpy+causallearn are available

req = json.loads(sys.argv[1])
op = req["op"]
if op == "gen_input":
    X = dgp._worker_gen_input(req["network"], req["n_samples"], req["seed"])
    buf = io.BytesIO(); np.save(buf, X)
    out = {"array_b64": base64.b64encode(buf.getvalue()).decode("ascii")}
elif op == "score":
    est = np.frombuffer(base64.b64decode(req["est_b64"]), dtype=np.int64).reshape(req["n_vars"], req["n_vars"])
    out = dgp._worker_score(req["network"], req["n_samples"], req["seed"], est)
else:
    raise SystemExit("unknown op: %r" % op)
sys.stdout.write("\n<<<DGP_RESULT>>>" + json.dumps(out) + "<<<END>>>\n")
sys.stdout.flush()
'''


def _run_in_conda_worker(req):
    conda = _find_conda_exe()
    if conda is None:
        raise RuntimeError(
            "causal-discovery-discrete host scorer needs pgmpy+causallearn but "
            "neither is importable in this Python and no conda executable was "
            "found to fall back to the '%s' env." % _CONDA_ENV
        )
    env = dict(os.environ)
    env["MLSB_HOLDOUT_DIR"] = str(_HERE.parent)
    env["MLSB_VENDORED_CAUSALLEARN"] = str(_VENDORED_CAUSALLEARN)
    cmd = [
        conda, "run", "--no-capture-output", "--name", _CONDA_ENV,
        "python", "-c", _WORKER_SRC, json.dumps(req),
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    out = proc.stdout
    start = out.find("<<<DGP_RESULT>>>")
    end = out.find("<<<END>>>")
    if start == -1 or end == -1:
        raise RuntimeError(
            "conda worker failed (rc=%s).\nSTDOUT tail:\n%s\nSTDERR tail:\n%s"
            % (proc.returncode, out[-2000:], proc.stderr[-2000:])
        )
    return json.loads(out[start + len("<<<DGP_RESULT>>>"):end])


# =====================================================================
# Public API used by mid_edit (input pre-generator) and parser (scorer).
# Keyed by the config test_cmd LABEL so callers never name the network.
# =====================================================================

def _spec(label):
    if label not in LABEL_SPEC:
        raise KeyError(f"unknown causal-discovery-discrete label: {label!r}")
    return LABEL_SPEC[label]


def gen_input(label, seed=42):
    """Agent-visible input ONLY: integer-encoded discrete samples X for a label.

    The true DAG is computed internally but deliberately NOT returned here, so
    the pre-generator that writes the agent's input file never persists the
    answer. Returns an int64 ndarray of shape (n_samples, n_variables).
    """
    network, n_samples = _spec(label)
    seed = int(seed)
    if _heavy_imports_available():
        return _worker_gen_input(network, n_samples, seed)
    res = _run_in_conda_worker(
        {"op": "gen_input", "network": network, "n_samples": n_samples, "seed": seed}
    )
    buf = io.BytesIO(base64.b64decode(res["array_b64"]))
    return np.load(buf)


def score(label, seed, est_matrix):
    """Host-side scorer: regenerate the true DAG for ``label`` and score the
    serialized est endpoint matrix. Returns the same metric dict / keys as the
    original ``compute_metrics``: shd, adj_precision, adj_recall, arrow_precision,
    arrow_recall.
    """
    network, n_samples = _spec(label)
    seed = int(seed)
    est_matrix = np.asarray(est_matrix, dtype=np.int64)
    n_vars = int(est_matrix.shape[0])
    if _heavy_imports_available():
        return _worker_score(network, n_samples, seed, est_matrix)
    est_b64 = base64.b64encode(
        np.ascontiguousarray(est_matrix, dtype=np.int64).tobytes()
    ).decode("ascii")
    return _run_in_conda_worker({
        "op": "score", "network": network, "n_samples": n_samples, "seed": seed,
        "est_b64": est_b64, "n_vars": n_vars,
    })
