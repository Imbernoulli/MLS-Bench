"""Task-specific output parser for meta-rl.

Training feedback: lines matching
    TRAIN_METRICS iteration=N avg_train_return=X.XXXX

Evaluation feedback: lines matching
    TEST_METRICS iteration=N meta_test_return=X.XXXX

Leaderboard metric: meta_test_return_{label} (from final evaluation, per environment).
"""

import re
import sys
from pathlib import Path

# Allow importing from mlsbench package when run standalone
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the meta-rl task."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        # Parse training metrics
        train_feedback = self._parse_train_metrics(raw_output)
        if train_feedback:
            feedback_parts.append(train_feedback)

        # Parse test metrics
        test_feedback, test_metrics = self._parse_test_metrics(raw_output, cmd_label)
        if test_feedback:
            feedback_parts.append(test_feedback)
        metrics.update(test_metrics)

        if feedback_parts:
            feedback = "\n".join(feedback_parts)
        else:
            feedback = raw_output

        return ParseResult(feedback=feedback, metrics=metrics)

    def _parse_train_metrics(self, output: str) -> str:
        """Extract TRAIN_METRICS lines and return a summary of the last few."""
        lines = []
        for line in output.splitlines():
            if line.strip().startswith("TRAIN_METRICS "):
                lines.append(line.strip())

        if not lines:
            return ""

        # Return last 5 training metric lines as feedback
        summary_lines = lines[-5:]
        return "Training metrics (last iterations):\n" + "\n".join(summary_lines)

    def _parse_test_metrics(self, output: str, cmd_label: str = "") -> tuple:
        """Extract TEST_METRICS lines and return feedback + label-qualified metrics.

        Expected format: TEST_METRICS iteration=N meta_test_return=X.XXXX
        """
        metrics: dict = {}
        feedback = ""

        for line in output.splitlines():
            match = re.search(
                r"TEST_METRICS\s+iteration=(\d+)\s+meta_test_return=([\d.-]+)",
                line,
            )
            if match:
                iteration = int(match.group(1))
                meta_test_return = float(match.group(2))
                # Label-qualify the metric key for multi-env leaderboard
                metric_key = "meta_test_return_" + cmd_label.replace("-", "_")
                metrics[metric_key] = meta_test_return
                feedback = (
                    f"Meta-test evaluation (iteration {iteration}):\n"
                    f"  Meta-test return: {meta_test_return:.4f}"
                )

        return feedback, metrics
