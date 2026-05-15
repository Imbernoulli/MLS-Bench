"""Output parser for ml-active-learning task.

Training feedback: lines matching
    TRAIN_METRICS round=R n_labeled=N accuracy=A

Evaluation feedback: lines matching
    TEST_METRICS accuracy=A auc=AUC

Leaderboard metrics: accuracy_{label}, auc_{label}
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the ml-active-learning (active learning query strategy) task."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        # Parse training metrics (learning curve progress)
        train_feedback = self._parse_train_metrics(raw_output, cmd_label)
        if train_feedback:
            feedback_parts.append(train_feedback)

        # Parse test metrics (final accuracy + AUC)
        test_feedback, test_metrics = self._parse_test_metrics(raw_output, cmd_label)
        if test_feedback:
            feedback_parts.append(test_feedback)
        metrics.update(test_metrics)

        if feedback_parts:
            feedback = "\n".join(feedback_parts)
        else:
            feedback = raw_output[-3000:]

        return ParseResult(feedback=feedback, metrics=metrics)

    def _parse_train_metrics(self, output: str, label: str) -> str:
        """Extract TRAIN_METRICS lines and return a summary."""
        lines = []
        for line in output.splitlines():
            if line.strip().startswith("TRAIN_METRICS"):
                lines.append(line.strip())

        if not lines:
            return ""

        # Show last 5 AL rounds as feedback
        summary_lines = lines[-5:]
        return f"Learning curve ({label}):\n" + "\n".join(summary_lines)

    def _parse_test_metrics(self, output: str, label: str) -> tuple[str, dict]:
        """Extract TEST_METRICS and return feedback + metrics dict."""
        metrics: dict = {}
        feedback = ""

        for line in output.splitlines():
            match = re.search(
                r"TEST_METRICS\s+accuracy=([\d.eE+-]+)\s+auc=([\d.eE+-]+)",
                line,
            )
            if match:
                accuracy = float(match.group(1))
                auc = float(match.group(2))
                metrics[f"accuracy_{label}"] = accuracy
                metrics[f"auc_{label}"] = auc
                feedback = (
                    f"Final metrics ({label}):\n"
                    f"  accuracy: {accuracy:.6f}\n"
                    f"  auc (learning curve): {auc:.6f}"
                )

        return feedback, metrics
