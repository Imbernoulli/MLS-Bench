"""Task-specific output parser for ml-anomaly-detection.

Handles combined train+eval output from anomaly detection:
- Training feedback: TRAIN_METRICS fold=F auroc=A f1=F
- Test feedback: TEST_METRICS auroc=A f1=F

Metrics are keyed by dataset label, e.g. auroc_cardio, f1_cardio.
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the ml-anomaly-detection task."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        train_feedback = self._parse_train_metrics(raw_output)
        if train_feedback:
            feedback_parts.append(train_feedback)

        eval_feedback, eval_metrics = self._parse_test_metrics(raw_output, cmd_label)
        if eval_feedback:
            feedback_parts.append(eval_feedback)
        metrics.update(eval_metrics)

        feedback = "\n".join(feedback_parts) if feedback_parts else raw_output[-3000:]
        return ParseResult(feedback=feedback, metrics=metrics)

    def _parse_train_metrics(self, output: str) -> str:
        lines = [l.strip() for l in output.splitlines() if l.strip().startswith("TRAIN_METRICS")]
        if not lines:
            return ""
        return "Cross-validation folds:\n" + "\n".join(lines[-5:])

    def _parse_test_metrics(self, output: str, cmd_label: str) -> tuple:
        metrics: dict = {}
        feedback = ""

        for line in output.splitlines():
            line = line.strip()
            if line.startswith("TEST_METRICS"):
                auroc_match = re.search(r"auroc=([\d.]+)", line)
                f1_match = re.search(r"f1=([\d.]+)", line)

                if auroc_match:
                    auroc = float(auroc_match.group(1))
                    key = f"auroc_{cmd_label}"
                    metrics[key] = auroc

                if f1_match:
                    f1 = float(f1_match.group(1))
                    key = f"f1_{cmd_label}"
                    metrics[key] = f1

                if auroc_match and f1_match:
                    feedback = (
                        f"Test results ({cmd_label}):\n"
                        f"  AUROC: {auroc:.4f}\n"
                        f"  F1:    {f1:.4f}"
                    )

        return feedback, metrics
