"""Host-side output parser / scorer for ml-anomaly-detection.

Runs in the harness process on the HOST — never inside the agent container. The
agent's program now only emits its test-set anomaly scores:

    ANOMALY_PRED env=<env> seed=<seed> n=<n> scores=<base64 float64>

The dataset identity (adbench file mapping), the train/test split, the test
labels, and the metrics live in ``holdout/ml-anomaly-detection/dgp.py`` — not
bind-mounted into the agent container. This parser regenerates y_test here and
scores AUROC + F1 with the same per-dataset keys (auroc_<env>, f1_<env>).
Honest results are identical to the pre-fix pipeline.
"""

import base64
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
# Locate the host-only DGP module. Native layout:
# <repo>/holdout/ml-anomaly-detection/dgp.py. Harbor layout: score_task.py loads this
# parser from a private copy of tests/meta/ with the held-out dgp.py staged
# next to it -- so also try the parser's own directory. Try both.
_HERE = Path(__file__).resolve().parent
for _cand in (PROJECT_ROOT / "holdout" / "ml-anomaly-detection", _HERE, _HERE / "holdout" / "ml-anomaly-detection"):
    if (_cand / "dgp.py").exists():
        sys.path.insert(0, str(_cand))
        break

from mlsbench.agent.parsers import OutputParser, ParseResult

import dgp  # noqa: E402  (host-only)

_PRED_RE = re.compile(
    r"ANOMALY_PRED\s+env=(\S+)\s+seed=(\d+)\s+n=(\d+)\s+scores=(\S+)"
)


class Parser(OutputParser):
    """Parser for ml-anomaly-detection (predict-then-score)."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        metrics: dict = {}
        feedback_parts = []

        for m in _PRED_RE.finditer(raw_output):
            env, seed_s, n_s, payload = m.groups()
            if env != cmd_label:
                continue
            seed = int(seed_s)
            try:
                scores = np.frombuffer(base64.b64decode(payload), dtype=np.float64)
            except Exception:
                continue
            if scores.shape[0] != int(n_s):
                continue
            y_test = dgp.truth(env, seed)
            if scores.shape[0] != y_test.shape[0]:
                continue
            res = dgp.score_predictions(scores, y_test)
            metrics[f"auroc_{cmd_label}"] = res["auroc"]
            metrics[f"f1_{cmd_label}"] = res["f1"]
            feedback_parts.append(f"auroc_{cmd_label}: {res['auroc']:.6f}")
            feedback_parts.append(f"f1_{cmd_label}: {res['f1']:.6f}")

        if not feedback_parts:
            return ParseResult(feedback=raw_output[-3000:], metrics={})
        return ParseResult(feedback="\n".join(feedback_parts), metrics=metrics)
