"""Host-side output parser / scorer for causal-observational-linear-non-gaussian.

Runs in the harness process on the HOST — never inside the agent container. The
agent's program now only emits its estimated adjacency matrix:

    CAUSAL_PRED graph_type=<g> n_nodes=<p> n_samples=<s> noise_type=<nt>
                er_prob=<e> sf_m=<m> seed=<seed> n=<p> adj=<base64 float64 matrix>

The LiNGAM data-generating process (which also produces the true adjacency
matrix) and the metrics live in
``holdout/causal-observational-linear-non-gaussian/dgp.py`` — not bind-mounted
into the agent container. This parser regenerates the true B here, reconstructs
the estimated B from the payload, and scores SHD + F1 + precision + recall with
the same per-setting keys (shd_<label>, f1_<label>, ...). Honest results are
identical to the pre-fix pipeline.
"""

import base64
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
# Locate the host-only DGP module. Native layout: <repo>/holdout/causal-observational-linear-non-gaussian/dgp.py.
# Harbor layout: score_task.py copies tests/meta/ to a private dir and loads
# parser.py from there, with the held-out dgp.py staged next to it. Try both.
_HERE = Path(__file__).resolve().parent
for _cand in (PROJECT_ROOT / "holdout" / "causal-observational-linear-non-gaussian", _HERE, _HERE / "holdout" / "causal-observational-linear-non-gaussian"):
    if (_cand / "dgp.py").exists():
        sys.path.insert(0, str(_cand))
        break

from mlsbench.agent.parsers import OutputParser, ParseResult

import dgp  # noqa: E402  (host-only)


_PRED_RE = re.compile(
    r"CAUSAL_PRED\s+"
    r"graph_type=(\S+)\s+"
    r"n_nodes=(\d+)\s+"
    r"n_samples=(\d+)\s+"
    r"noise_type=(\S+)\s+"
    r"er_prob=(\S+)\s+"
    r"sf_m=(\d+)\s+"
    r"seed=(\d+)\s+"
    r"n=(\d+)\s+"
    r"adj=(\S+)"
)


class Parser(OutputParser):
    """Parse adjacency predictions emitted by bench/run_eval.py and score host-side."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        metrics: dict = {}
        feedback_parts = []

        for mt in _PRED_RE.finditer(raw_output):
            (graph_type, n_nodes_s, n_samples_s, noise_type, er_prob_s, sf_m_s,
             seed_s, n_s, payload) = mt.groups()
            n_nodes = int(n_nodes_s)
            try:
                B_est = np.frombuffer(base64.b64decode(payload), dtype=np.float64)
            except Exception:
                continue
            if B_est.shape[0] != n_nodes * n_nodes:
                continue
            B_est = B_est.reshape(n_nodes, n_nodes)

            B_true = dgp.truth(
                graph_type=graph_type,
                n_nodes=n_nodes,
                n_samples=int(n_samples_s),
                noise_type=noise_type,
                er_prob=float(er_prob_s),
                sf_m=int(sf_m_s),
                seed=int(seed_s),
            )
            m = dgp.compute_metrics(B_est, B_true)

            metrics[f"shd_{cmd_label}"]       = m["shd"]
            metrics[f"f1_{cmd_label}"]        = m["f1"]
            metrics[f"precision_{cmd_label}"] = m["precision"]
            metrics[f"recall_{cmd_label}"]    = m["recall"]

            feedback_parts.append(
                f"Results ({cmd_label}):\n"
                f"  SHD={m['shd']}  F1={m['f1']:.4f}  "
                f"Precision={m['precision']:.4f}  Recall={m['recall']:.4f}"
            )

        feedback = "\n".join(feedback_parts) if feedback_parts else raw_output
        return ParseResult(feedback=feedback, metrics=metrics)
