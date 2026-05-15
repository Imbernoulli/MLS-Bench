"""Output parser for opt-variance-reduction.

Handles output from custom_vr.py:
- Training feedback: TRAIN_METRICS: epoch=N avg_loss=L time=Ts grad_comps=G
- Evaluation metrics: EVAL_METRICS: epoch=N test_accuracy=A / test_mse=M
- Final metrics: TEST_METRICS: best_<metric>=V final_<metric>=V total_grad_comps=G
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


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

        test_feedback, test_metrics = self._parse_test_metrics(
            raw_output, cmd_label)
        if test_feedback:
            feedback_parts.append(test_feedback)
        metrics.update(test_metrics)

        feedback = "\n".join(feedback_parts) if feedback_parts else raw_output[-3000:]
        return ParseResult(feedback=feedback, metrics=metrics)

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
