"""Host-side output parser / scorer for ml-ensemble-boosting.

Runs in the harness process on the HOST -- never inside the agent container. The
agent's program now only emits its boosting ensemble's test predictions:

    BOOST_PRED env=<env> seed=<seed> n=<n> preds=<base64 float64>

The dataset identity, the deterministic train/test split, the test labels, and
the metric live in ``holdout/ml-ensemble-boosting/dgp.py`` -- not bind-mounted
into the agent container (so the agent neither knows the eval datasets nor can
reload the public data, replay the split, and recover y_test). This parser
regenerates y_test here and scores the same metric with the same per-dataset
keys (test_accuracy_<env> for classification, test_rmse_<env> for regression).
Honest results are identical to the pre-fix pipeline.
"""

import base64
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
_HOLDOUT_DIR = PROJECT_ROOT / "holdout" / "ml-ensemble-boosting"
sys.path.insert(0, str(_HOLDOUT_DIR))

from mlsbench.agent.parsers import OutputParser, ParseResult

import dgp  # noqa: E402  (host-only)

_PRED_RE = re.compile(
    r"BOOST_PRED\s+env=(\S+)\s+seed=(\d+)\s+n=(\d+)\s+preds=(\S+)"
)


class Parser(OutputParser):
    """Parser for ml-ensemble-boosting (predict-then-score)."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        train_feedback = self._parse_train_metrics(raw_output)
        if train_feedback:
            feedback_parts.append(train_feedback)

        for m in _PRED_RE.finditer(raw_output):
            env, seed_s, n_s, payload = m.groups()
            if env != cmd_label:
                continue
            seed = int(seed_s)
            try:
                preds = np.frombuffer(base64.b64decode(payload), dtype=np.float64)
            except Exception:
                continue
            if preds.shape[0] != int(n_s):
                continue
            y_test = dgp.truth(env, seed)
            if preds.shape[0] != y_test.shape[0]:
                continue
            scores = dgp.score_predictions(env, preds, y_test)
            parts = []
            for k, v in scores.items():
                mk = f"{k}_{cmd_label}"
                metrics[mk] = float(v)
                parts.append(f"{mk}={float(v):.4f}")
            if parts:
                feedback_parts.append(
                    f"Final metrics ({cmd_label}): " + ", ".join(parts)
                )

        if not feedback_parts:
            return ParseResult(feedback=raw_output[-3000:], metrics=metrics)
        return ParseResult(feedback="\n".join(feedback_parts), metrics=metrics)

    def _parse_train_metrics(self, output: str) -> str:
        lines = [
            l.strip()
            for l in output.splitlines()
            if l.strip().startswith("TRAIN_METRICS:")
        ]
        if not lines:
            return ""
        return "Training progress (last rounds):\n" + "\n".join(lines[-5:])
