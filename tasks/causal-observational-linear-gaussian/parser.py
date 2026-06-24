"""Host-side output parser / scorer for causal-observational-linear-gaussian.

Runs in the harness process on the HOST — never inside the agent container. The
agent's program now only emits its estimated CPDAG as a causallearn endpoint
matrix:

    CAUSAL_PRED graph_type=<g> n_nodes=<p> n_samples=<s> er_prob=<e> sf_m=<m>
                noise_scale=<ns> seed=<seed> n=<p> adj=<base64 int64 matrix>

The data-generating process (which also produces the true DAG) and the CPDAG
metrics live in ``holdout/causal-observational-linear-gaussian/dgp.py`` — not
bind-mounted into the agent container. This parser regenerates the true DAG here,
reconstructs the estimated GeneralGraph from the endpoint matrix, and scores SHD
+ adjacency/arrow precision/recall with the same per-setting keys
(shd_<label>, adj_precision_<label>, ...). Honest results are identical to the
pre-fix pipeline.
"""
import base64
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
_HOLDOUT_DIR = PROJECT_ROOT / "holdout" / "causal-observational-linear-gaussian"
sys.path.insert(0, str(_HOLDOUT_DIR))

from mlsbench.agent.parsers import OutputParser, ParseResult

import dgp  # noqa: E402  (host-only)

from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode


_PRED_RE = re.compile(
    r"CAUSAL_PRED\s+"
    r"graph_type=(\S+)\s+"
    r"n_nodes=(\d+)\s+"
    r"n_samples=(\d+)\s+"
    r"er_prob=(\S+)\s+"
    r"sf_m=(\d+)\s+"
    r"noise_scale=(\S+)\s+"
    r"seed=(\d+)\s+"
    r"n=(\d+)\s+"
    r"adj=(\S+)"
)


def _graph_from_matrix(mat):
    """Rebuild a causallearn GeneralGraph from its endpoint matrix (lossless)."""
    n = mat.shape[0]
    nodes = [GraphNode(f"X{i + 1}") for i in range(n)]
    g = GeneralGraph(nodes)
    g.graph = np.asarray(mat, dtype=int)
    return g


class Parser(OutputParser):
    """Parse CPDAG predictions emitted by bench/run_eval.py and score host-side."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        metrics: dict = {}
        feedback_parts = []

        for m in _PRED_RE.finditer(raw_output):
            (graph_type, n_nodes_s, n_samples_s, er_prob_s, sf_m_s,
             noise_scale_s, seed_s, n_s, payload) = m.groups()
            n_nodes = int(n_nodes_s)
            try:
                mat = np.frombuffer(base64.b64decode(payload), dtype=np.int64)
            except Exception:
                continue
            if mat.shape[0] != n_nodes * n_nodes:
                continue
            mat = mat.reshape(n_nodes, n_nodes)

            true_dag = dgp.truth(
                graph_type=graph_type,
                n_nodes=n_nodes,
                n_samples=int(n_samples_s),
                er_prob=float(er_prob_s),
                sf_m=int(sf_m_s),
                noise_scale=float(noise_scale_s),
                seed=int(seed_s),
            )
            est_graph = _graph_from_matrix(mat)
            res = dgp.compute_metrics(est_graph, true_dag)

            metrics[f"shd_{cmd_label}"] = res["shd"]
            metrics[f"adj_precision_{cmd_label}"] = res["adj_precision"]
            metrics[f"adj_recall_{cmd_label}"] = res["adj_recall"]
            metrics[f"arrow_precision_{cmd_label}"] = res["arrow_precision"]
            metrics[f"arrow_recall_{cmd_label}"] = res["arrow_recall"]

            feedback_parts.append(
                f"Results ({cmd_label}):\n"
                f"  SHD={res['shd']}  "
                f"AdjP={res['adj_precision']:.4f} AdjR={res['adj_recall']:.4f}  "
                f"ArrowP={res['arrow_precision']:.4f} ArrowR={res['arrow_recall']:.4f}"
            )

        feedback = "\n".join(feedback_parts) if feedback_parts else raw_output
        return ParseResult(feedback=feedback, metrics=metrics)
