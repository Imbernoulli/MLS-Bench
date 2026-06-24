"""Host-side output parser / scorer for ml-symbolic-regression.

Runs in the harness process on the HOST — never inside the agent container. The
agent's program drives the GP search on the provided (X_train, y_train) samples
and then only emits its best evolved expression's predictions on the held-out
test inputs:

    SR_PRED task=<opaque> seed=<s> n=<n> preds=<base64 float64>

The benchmark TARGET FUNCTIONS, the data generator, and the R2 metric live in
``holdout/ml-symbolic-regression/dgp.py`` — a module that is NOT bind-mounted
into the agent container. This parser regenerates the held-out test labels here,
host-side, keyed by the command's benchmark label (cmd_label), and computes the
official ``test_r2_<benchmark>`` with the SAME R2 formula as before. Honest
results are therefore identical to the pre-fix pipeline, while the agent's
process can never read the closed-form target, the benchmark name, or y_test.

TRAIN_METRICS / TEST_METRICS lines (train-side fit summaries) are still
surfaced as feedback, but they carry no held-out information.
"""

import base64
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
# Host-only scoring module (never inside the agent container).
_HOLDOUT_DIR = PROJECT_ROOT / "holdout" / "ml-symbolic-regression"
sys.path.insert(0, str(_HOLDOUT_DIR))

from mlsbench.agent.parsers import OutputParser, ParseResult

import dgp  # noqa: E402  (host-only)


_PRED_RE = re.compile(
    r"SR_PRED\s+task=(\S+)\s+seed=(\d+)\s+n=(\d+)\s+preds=(\S+)"
)


class Parser(OutputParser):
    """Parser for the ml-symbolic-regression task (predict-then-score)."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        # Train-side feedback (no held-out information).
        train_feedback = self._parse_train_metrics(raw_output)
        if train_feedback:
            feedback_parts.append(train_feedback)
        test_feedback = self._parse_test_metrics(raw_output)
        if test_feedback:
            feedback_parts.append(test_feedback)

        # Official scoring: host-side R2 from emitted predictions.
        scored = False
        for m in _PRED_RE.finditer(raw_output):
            _task, seed_s, n_s, payload = m.groups()
            seed = int(seed_s)
            try:
                preds = np.frombuffer(base64.b64decode(payload), dtype=np.float64)
            except Exception:
                continue
            if preds.shape[0] != int(n_s):
                continue
            try:
                y_test = dgp.truth(cmd_label, seed)
            except KeyError:
                continue
            if preds.shape[0] != y_test.shape[0]:
                continue
            test_r2 = float(dgp.r2_score(y_test, preds))
            metric_key = "test_r2_" + cmd_label.replace("-", "_")
            metrics[metric_key] = test_r2
            feedback_parts.append(
                f"Test results ({cmd_label}):\n  R² = {test_r2:.6f}"
            )
            scored = True

        if not scored and not feedback_parts:
            # No scorable predictions — surface a tail of the raw output to help
            # the agent debug (no ground truth is ever included here).
            return ParseResult(feedback=raw_output[-3000:], metrics=metrics)

        feedback = "\n".join(feedback_parts) if feedback_parts else raw_output
        return ParseResult(feedback=feedback, metrics=metrics)

    def _parse_train_metrics(self, output: str) -> str:
        """Extract TRAIN_METRICS lines and return a summary of the last few."""
        lines = []
        for line in output.splitlines():
            if line.strip().startswith("TRAIN_METRICS "):
                lines.append(line.strip())
        if not lines:
            return ""
        summary_lines = lines[-5:]
        return "Training metrics (last generations):\n" + "\n".join(summary_lines)

    def _parse_test_metrics(self, output: str) -> str:
        """Surface the train-side fit summary (train_r2, size, expression)."""
        for line in output.splitlines():
            s = line.strip()
            if not s.startswith("TEST_METRICS "):
                continue
            train_match = re.search(r"train_r2=(-?[\d.]+(?:e[+-]?\d+)?)", s)
            expr_match = re.search(r'expression="([^"]*)"', s)
            parts = [f"Fit summary:\n  {s}"]
            if train_match:
                parts.append(f"  train R² = {float(train_match.group(1)):.6f}")
            if expr_match:
                parts.append(f"  Expression: {expr_match.group(1)}")
            return "\n".join(parts)
        return ""
