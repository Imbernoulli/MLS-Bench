"""Host-side output parser / scorer for causal-discovery-discrete.

Runs in the harness process on the HOST — never inside the agent container. The
agent's program now only emits its estimated graph as an integer endpoint
adjacency matrix:

    CAUSAL_PRED label=<label> seed=<seed> n=<n> adj=<base64 int64 matrix>

The evaluation network identity, the ground-truth DAG construction
(``get_example_model``), and the metric (SHD / adjacency- and arrow-
precision/recall via causallearn) live in
``holdout/causal-discovery-discrete/dgp.py`` — a module that is NOT bind-mounted
into the agent container. This parser regenerates the true CPDAG there,
reconstructs the estimated graph from the serialized endpoint matrix, and scores
it with the SAME metric keys as before (shd_<label>, adj_precision_<label>,
adj_recall_<label>, arrow_precision_<label>, arrow_recall_<label>). Honest
results are identical to the pre-fix pipeline.
"""
import base64
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
# Host-only scoring module (never inside the agent container).
# Locate the host-only DGP module. Native layout:
# <repo>/holdout/causal-discovery-discrete/dgp.py. Harbor layout: score_task.py loads this
# parser from a private copy of tests/meta/ with the held-out dgp.py staged
# next to it -- so also try the parser's own directory. Try both.
_HERE = Path(__file__).resolve().parent
for _cand in (PROJECT_ROOT / "holdout" / "causal-discovery-discrete", _HERE, _HERE / "holdout" / "causal-discovery-discrete"):
    if (_cand / "dgp.py").exists():
        sys.path.insert(0, str(_cand))
        break

from mlsbench.agent.parsers import OutputParser, ParseResult

import dgp  # noqa: E402  (host-only, from holdout/causal-discovery-discrete/)


# The agent program no longer echoes the eval-case identity on the CAUSAL_PRED
# line (it only ever saw an opaque token anyway); the host-side scorer already
# knows which case it grades from ``cmd_label``. Each eval run emits one line.
_PRED_RE = re.compile(
    r"CAUSAL_PRED\s+seed=(\d+)\s+n=(\d+)\s+adj=(\S+)"
)


class Parser(OutputParser):
    """Parse CAUSAL_PRED, regenerate the true CPDAG, and score it."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        metrics: dict = {}
        feedback_parts = []

        label = cmd_label
        for m in _PRED_RE.finditer(raw_output):
            seed_s, n_s, payload = m.groups()
            seed = int(seed_s)
            n = int(n_s)
            try:
                adj = np.frombuffer(base64.b64decode(payload), dtype=np.int64)
            except Exception:
                continue
            if adj.shape[0] != n * n:
                continue
            adj = adj.reshape(n, n)

            res = dgp.score(label, seed, adj)

            shd = res["shd"]
            adj_precision = res["adj_precision"]
            adj_recall = res["adj_recall"]
            arrow_precision = res["arrow_precision"]
            arrow_recall = res["arrow_recall"]

            metrics[f"shd_{cmd_label}"] = shd
            metrics[f"adj_precision_{cmd_label}"] = adj_precision
            metrics[f"adj_recall_{cmd_label}"] = adj_recall
            metrics[f"arrow_precision_{cmd_label}"] = arrow_precision
            metrics[f"arrow_recall_{cmd_label}"] = arrow_recall

            feedback_parts.append(
                f"Results ({cmd_label}):\n"
                f"  SHD={shd}  "
                f"AdjP={adj_precision:.4f} AdjR={adj_recall:.4f}  "
                f"ArrowP={arrow_precision:.4f} ArrowR={arrow_recall:.4f}"
            )

        feedback = "\n".join(feedback_parts) if feedback_parts else raw_output
        return ParseResult(feedback=feedback, metrics=metrics)
