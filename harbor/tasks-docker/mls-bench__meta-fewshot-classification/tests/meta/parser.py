"""Task-specific output parser for meta-fewshot-classification.

Handles combined train+eval output from few-shot classification:
- Training feedback: TRAIN_METRICS epoch=E train_loss=L val_acc=A
- Test feedback: TEST_METRICS accuracy=A

Metrics are keyed by dataset label, e.g. accuracy_mini_imagenet.
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the meta-fewshot-classification task."""

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

        feedback = "\n".join(feedback_parts) if feedback_parts else raw_output
        return ParseResult(feedback=feedback, metrics=metrics)

    def _parse_train_metrics(self, output: str) -> str:
        lines = [l.strip() for l in output.splitlines() if l.strip().startswith("TRAIN_METRICS ")]
        if not lines:
            return ""
        return "Training metrics (last epochs):\n" + "\n".join(lines[-5:])

    def _parse_test_metrics(self, output: str, cmd_label: str) -> tuple[str, dict]:
        metrics: dict = {}
        feedback = ""

        for line in output.splitlines():
            match = re.search(r"TEST_METRICS\s+accuracy=([\d.]+)", line)
            if match:
                accuracy = float(match.group(1))
                metric_key = "accuracy_" + cmd_label.replace("-", "_")
                metrics[metric_key] = accuracy
                feedback = (
                    f"Test results ({cmd_label}):\n"
                    f"  Accuracy: {accuracy:.4f} ({100 * accuracy:.2f}%)"
                )

        return feedback, metrics
