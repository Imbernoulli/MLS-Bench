"""Task-specific output parser for jepa-regularizer.

Training feedback: lines matching
    TRAIN_METRICS: epoch=E | ... | val_acc=XX.XX | ...

Final metric: line matching
    TEST_METRICS: val_acc=XX.XX

Leaderboard metric: val_acc_cifar10 (linear probe accuracy on CIFAR-10).
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the jepa-regularizer task."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        train_feedback = self._parse_train_metrics(raw_output)
        if train_feedback:
            feedback_parts.append(train_feedback)

        test_feedback, test_metrics = self._parse_test_metrics(raw_output, cmd_label)
        if test_feedback:
            feedback_parts.append(test_feedback)
        metrics.update(test_metrics)

        feedback = "\n".join(feedback_parts) if feedback_parts else raw_output
        return ParseResult(feedback=feedback, metrics=metrics)

    def _parse_train_metrics(self, output: str) -> str:
        """Extract TRAIN_METRICS lines and return a summary of the last few."""
        lines = [
            l.strip()
            for l in output.splitlines()
            if l.strip().startswith("TRAIN_METRICS:")
        ]
        if not lines:
            return ""
        return "Training metrics (last steps):\n" + "\n".join(lines[-5:])

    def _parse_test_metrics(
        self, output: str, cmd_label: str
    ) -> tuple[str, dict]:
        """Extract TEST_METRICS line and return feedback + metrics.

        Expected format: TEST_METRICS: val_acc=XX.XX
        """
        metrics: dict = {}
        feedback = ""

        for line in output.splitlines():
            match = re.search(
                r"TEST_METRICS:\s*val_acc=([\d.]+)", line
            )
            if match:
                val_acc = float(match.group(1))
                metric_key = f"val_acc_{cmd_label}"
                metrics[metric_key] = val_acc
                feedback = f"Final validation accuracy ({cmd_label}): {val_acc:.2f}%"

        return feedback, metrics
