"""Host-side output parser / scorer for ml-clustering-algorithm.

Runs in the harness process on the HOST — never inside the agent container. The
agent's program now only emits its cluster assignments:

    CLUSTER_PRED env=<env> seed=<seed> n=<n> labels=<base64 int64>

The dataset identity, the true cluster labels, and the metrics live in
``holdout/ml-clustering-algorithm/dgp.py`` — not bind-mounted into the agent
container (so the agent neither knows the eval datasets nor can regenerate the
true labels). This parser regenerates the truth here and scores ARI/NMI/
silhouette with the same per-dataset keys (ari_<env>, nmi_<env>,
silhouette_<env>). Honest results are identical to the pre-fix pipeline.
"""

import base64
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
# Locate the host-only DGP module. Native layout:
# <repo>/holdout/ml-clustering-algorithm/dgp.py. Harbor layout: score_task.py loads this
# parser from a private copy of tests/meta/ with the held-out dgp.py staged
# next to it -- so also try the parser's own directory. Try both.
_HERE = Path(__file__).resolve().parent
for _cand in (PROJECT_ROOT / "holdout" / "ml-clustering-algorithm", _HERE, _HERE / "holdout" / "ml-clustering-algorithm"):
    if (_cand / "dgp.py").exists():
        sys.path.insert(0, str(_cand))
        break

from mlsbench.agent.parsers import OutputParser, ParseResult

import dgp  # noqa: E402  (host-only)

_PRED_RE = re.compile(
    r"CLUSTER_PRED\s+env=(\S+)\s+seed=(\d+)\s+n=(\d+)\s+labels=(\S+)"
)


class Parser(OutputParser):
    """Parser for ml-clustering-algorithm (predict-then-score)."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        metrics: dict = {}
        feedback_parts = []

        for m in _PRED_RE.finditer(raw_output):
            env, seed_s, n_s, payload = m.groups()
            if env != cmd_label:
                continue
            seed = int(seed_s)
            try:
                labels = np.frombuffer(base64.b64decode(payload), dtype=np.int64)
            except Exception:
                continue
            if labels.shape[0] != int(n_s):
                continue
            X, y_true = dgp.truth(env, seed)
            if labels.shape[0] != X.shape[0]:
                continue
            scores = dgp.evaluate_clustering(X, y_true, labels)
            for k, v in scores.items():
                mk = f"{k}_{cmd_label}"
                metrics[mk] = float(v)
                feedback_parts.append(f"{mk}: {float(v):.6f}")

        if not feedback_parts:
            return ParseResult(feedback=raw_output[-3000:], metrics={})
        return ParseResult(feedback="\n".join(feedback_parts), metrics=metrics)
