"""Task-specific output parser for sr-symbolic-regression.

Handles combined train+test output from GP symbolic regression:

Training feedback: lines matching
    TRAIN_METRICS generation=N best_fitness=F avg_fitness=F best_size=S train_r2=R

Test feedback: lines matching
    TEST_METRICS r2=R rmse=E train_r2=R size=S expression="..."

Metrics are keyed by benchmark name, e.g. test_r2_nguyen7.
"""

import re
import sys
from pathlib import Path

# Allow importing from mlsbench package when run standalone
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the sr-symbolic-regression task."""

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

        summary_lines = lines[-5:]
        return "Training metrics (last generations):\n" + "\n".join(summary_lines)

    def _parse_test_metrics(self, output: str, cmd_label: str) -> tuple[str, dict]:
        """Extract TEST_METRICS line and return feedback + metrics.

        Expected format: TEST_METRICS r2=R rmse=E train_r2=R size=S expression="..."
        """
        metrics: dict = {}
        feedback = ""

        for line in output.splitlines():
            if not line.strip().startswith("TEST_METRICS "):
                continue

            # Extract r2
            r2_match = re.search(r"r2=(-?[\d.]+(?:e[+-]?\d+)?)", line)
            if r2_match:
                r2_val = float(r2_match.group(1))
                metric_key = "test_r2_" + cmd_label.replace("-", "_")
                metrics[metric_key] = r2_val

            # Extract rmse
            rmse_match = re.search(r"rmse=([\d.]+(?:e[+-]?\d+)?)", line)

            # Extract expression
            expr_match = re.search(r'expression="([^"]*)"', line)

            feedback = f"Test results ({cmd_label}):\n  {line.strip()}"
            if r2_match:
                feedback += f"\n  R² = {float(r2_match.group(1)):.6f}"
            if rmse_match:
                feedback += f"\n  RMSE = {float(rmse_match.group(1)):.6f}"
            if expr_match:
                feedback += f"\n  Expression: {expr_match.group(1)}"

        return feedback, metrics
