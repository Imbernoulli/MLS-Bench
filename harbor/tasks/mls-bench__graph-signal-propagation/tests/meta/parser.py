"""Task-specific output parser for graph-signal-propagation.

Handles combined train+eval output from graph node classification:
- Training feedback: TRAIN_METRICS run=R epoch=E train_loss=L val_acc=A test_acc=A
- Test feedback: TEST_METRICS accuracy=A std=S

Metrics are keyed by dataset label, e.g. accuracy_cora, accuracy_texas.
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the graph-signal-propagation task."""

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
        lines = [l.strip() for l in output.splitlines()
                 if l.strip().startswith("TRAIN_METRICS")]
        if not lines:
            return ""
        # Show last 5 training lines (final run results)
        return "Training metrics (last entries):\n" + "\n".join(lines[-5:])

    def _parse_test_metrics(self, output: str, cmd_label: str) -> tuple:
        metrics: dict = {}
        feedback = ""

        for line in output.splitlines():
            match = re.search(
                r"TEST_METRICS\s+accuracy=([\d.]+)(?:\s+std=([\d.]+))?", line
            )
            if match:
                accuracy = float(match.group(1))
                metric_key = "accuracy_" + cmd_label.replace("-", "_")
                metrics[metric_key] = accuracy
                feedback_str = f"  Accuracy: {accuracy:.4f} ({100 * accuracy:.2f}%)"
                if match.group(2):
                    std = float(match.group(2))
                    std_key = "std_" + cmd_label.replace("-", "_")
                    metrics[std_key] = std
                    feedback_str += f" +/- {100 * std:.2f}%"
                feedback = f"Test results ({cmd_label}):\n{feedback_str}"

        return feedback, metrics
