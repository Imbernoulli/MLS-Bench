"""Output parser for opt-variance-reduction.

Handles output from custom_vr.py / vr_driver.py:
- Training feedback: TRAIN_METRICS: epoch=N avg_loss=L time=Ts grad_comps=G
- Evaluation metrics: EVAL_METRICS: epoch=N test_accuracy=A / test_mse=M
- In-container final (logistic/mlp): TEST_METRICS: best_<m>=V final_<m>=V total_grad_comps=G
- Host-scored final (synthetic 'conditioned', whose test labels never enter the
  container): the driver emits raw test predictions per eval epoch
  ``VR_PRED epoch=E shape=NxK preds=<b64 float32>`` plus
  ``VR_TRAJ problem=P seed=S target_metric=test_mse higher_is_better=0 total_grad_comps=G``;
  this parser regenerates y_test via the host-only dgp and computes the SAME
  best_/final_ metric, so honest numbers are unchanged while the test labels are
  never present in the agent's process.
"""

import base64
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


def _load_dgp():
    """Import the host-only DGP, supporting both native and Harbor layouts.

    Native: ``<repo>/holdout/optimization-variance-reduction/dgp.py``.
    Harbor:  ``dgp.py`` staged next to this parser (tests/meta/).
    """
    here = Path(__file__).resolve().parent
    for cand in (PROJECT_ROOT / "holdout" / "optimization-variance-reduction", here):
        if (cand / "dgp.py").is_file():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            import dgp  # noqa: E402
            return dgp
    raise ImportError("opt-variance-reduction dgp.py not found (native or Harbor)")


_VR_PRED_RE = re.compile(
    r"VR_PRED\s+epoch=(\d+)\s+shape=(\S+)\s+preds=(\S+)"
)
_VR_TRAJ_RE = re.compile(
    r"VR_TRAJ\s+problem=(\S+)\s+seed=(\d+)\s+target_metric=(\S+)\s+"
    r"higher_is_better=(\d+)\s+total_grad_comps=(\d+)"
)


class Parser(OutputParser):
    """Parser for variance reduction benchmark."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        train_feedback = self._parse_train_metrics(raw_output)
        if train_feedback:
            feedback_parts.append(train_feedback)

        eval_feedback = self._parse_eval_metrics(raw_output)
        if eval_feedback:
            feedback_parts.append(eval_feedback)

        # Host-scored path (synthetic 'conditioned'): the held-out test labels
        # are not in the container, so regenerate them here and score the emitted
        # predictions. Falls through to the in-container TEST_METRICS path when no
        # VR_TRAJ line is present (logistic/mlp).
        traj = _VR_TRAJ_RE.search(raw_output)
        if traj is not None:
            host_feedback, host_metrics = self._parse_host_scored(
                raw_output, cmd_label, traj)
            if host_feedback:
                feedback_parts.append(host_feedback)
            metrics.update(host_metrics)
        else:
            test_feedback, test_metrics = self._parse_test_metrics(
                raw_output, cmd_label)
            if test_feedback:
                feedback_parts.append(test_feedback)
            metrics.update(test_metrics)

        feedback = "\n".join(feedback_parts) if feedback_parts else raw_output[-3000:]
        return ParseResult(feedback=feedback, metrics=metrics)

    def _parse_host_scored(self, output: str, cmd_label: str, traj) -> tuple:
        problem, seed_s, target_metric, hib_s, gc_s = traj.groups()
        seed = int(seed_s)
        higher_is_better = bool(int(hib_s))
        total_grad_comps = int(gc_s)

        dgp = _load_dgp()
        _X_test, y_test = dgp.truth(problem, seed)
        y_test = np.asarray(y_test, dtype=np.float64)
        n_test = y_test.shape[0]

        # test_mse matches the in-container evaluate(): sum of squared errors over
        # all elements divided by the number of test samples.
        vals = []
        for m in _VR_PRED_RE.finditer(output):
            _epoch, shape_s, payload = m.groups()
            try:
                preds = np.frombuffer(
                    base64.b64decode(payload), dtype=np.float32
                ).astype(np.float64)
            except Exception:
                continue
            dims = [int(s) for s in shape_s.split("x")]
            preds = preds.reshape(dims)
            if preds.shape[0] != n_test:
                continue
            yt = y_test.reshape(preds.shape)
            vals.append(float(np.sum((preds - yt) ** 2) / n_test))

        metrics: dict = {f"total_grad_comps_{cmd_label}": float(total_grad_comps)}
        feedback = ""
        if vals:
            best = max(vals) if higher_is_better else min(vals)
            final = vals[-1]
            metrics[f"best_{target_metric}_{cmd_label}"] = float(best)
            metrics[f"final_{target_metric}_{cmd_label}"] = float(final)
            parts = [f"{k}={v:.4f}" for k, v in metrics.items()]
            feedback = f"Final metrics ({cmd_label}): " + ", ".join(parts)
        return feedback, metrics

    def _parse_train_metrics(self, output: str) -> str:
        lines = [l.strip() for l in output.splitlines()
                 if l.strip().startswith("TRAIN_METRICS:")]
        if not lines:
            return ""
        return "Training progress (last 5 epochs):\n" + "\n".join(lines[-5:])

    def _parse_eval_metrics(self, output: str) -> str:
        lines = [l.strip() for l in output.splitlines()
                 if l.strip().startswith("EVAL_METRICS:")]
        if not lines:
            return ""
        return "Evaluation progress (last 5 evals):\n" + "\n".join(lines[-5:])

    def _parse_test_metrics(self, output: str, cmd_label: str) -> tuple:
        metrics: dict = {}
        feedback = ""

        for line in output.splitlines():
            if "TEST_METRICS:" not in line:
                continue
            pairs = re.findall(
                r"(\w+)=([\d.]+(?:e[+-]?\d+)?|nan|inf|-inf)",
                line, re.IGNORECASE
            )
            for key, raw in pairs:
                val = float(raw.lower())
                metric_key = f"{key}_{cmd_label}"
                metrics[metric_key] = val
            if metrics:
                parts = [f"{k}={v:.4f}" for k, v in metrics.items()]
                feedback = f"Final metrics ({cmd_label}): " + ", ".join(parts)

        return feedback, metrics
