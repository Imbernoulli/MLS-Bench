"""Task-specific output parser for automl-nas-search.
Handles combined search+eval output from the NAS optimizer:
- Training feedback: TRAIN_METRICS epoch=E best_val_acc=A queries=Q
- Test feedback: TEST_METRICS test_accuracy=A
Metrics are keyed by dataset label, e.g. test_accuracy_CIFAR-10.
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the automl-nas-search task."""

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
        return "Search progress (last epochs):\n" + "\n".join(lines[-5:])

    def _parse_eval_metrics(self, output: str, cmd_label: str) -> tuple[str, dict]:
        metrics: dict = {}
        feedback = ""

        for line in output.splitlines():
            match = re.search(r"TEST_METRICS\s+test_accuracy=([\d.eE+-]+)", line)
            if match:
                test_acc = float(match.group(1))
                metrics[f"test_accuracy_{cmd_label}"] = test_acc
                feedback = (
                    f"Test results ({cmd_label}):\n"
                    f"  Test accuracy: {test_acc:.4f}"
                )

        return feedback, metrics
