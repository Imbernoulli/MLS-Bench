"""Task-specific output parser for ts-anomaly-detection.
Handles combined train+eval output from TSLib anomaly detection:
- Training feedback: TRAIN_METRICS epoch=E train_loss=L vali_loss=V test_loss=T
- Test feedback: Accuracy : X.XXXX, Precision : X.XXXX, Recall : X.XXXX, F-score : X.XXXX
Metrics keyed by dataset label, e.g. f_score_PSM.
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the ts-anomaly-detection task."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        train_feedback = self._parse_train_metrics(raw_output)
        if train_feedback:
            feedback_parts.append(train_feedback)

        eval_feedback, eval_metrics = self._parse_eval_metrics(raw_output, cmd_label)
        if eval_feedback:
            feedback_parts.append(eval_feedback)
        metrics.update(eval_metrics)

        feedback = "\n".join(feedback_parts) if feedback_parts else raw_output
        return ParseResult(feedback=feedback, metrics=metrics)

    def _parse_train_metrics(self, output: str) -> str:
        lines = [l.strip() for l in output.splitlines() if l.strip().startswith("TRAIN_METRICS ")]
        if not lines:
            return ""
        return "Training metrics (last epochs):\n" + "\n".join(lines[-5:])

    def _parse_eval_metrics(self, output: str, cmd_label: str) -> tuple[str, dict]:
        metrics: dict = {}
        feedback = ""

        for line in output.splitlines():
            match = re.search(
                r"Accuracy\s*:\s*([\d.]+).*Precision\s*:\s*([\d.]+).*Recall\s*:\s*([\d.]+).*F-score\s*:\s*([\d.]+)",
                line
            )
            if match:
                accuracy = float(match.group(1))
                precision = float(match.group(2))
                recall = float(match.group(3))
                f_score = float(match.group(4))
                metrics[f"f_score_{cmd_label}"] = f_score
                metrics[f"precision_{cmd_label}"] = precision
                metrics[f"recall_{cmd_label}"] = recall
                feedback = (
                    f"Test results ({cmd_label}):\n"
                    f"  F-score: {f_score:.4f}, Precision: {precision:.4f}, "
                    f"Recall: {recall:.4f}, Accuracy: {accuracy:.4f}"
                )

        return feedback, metrics
