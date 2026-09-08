"""Task-specific output parser for jepa-planning.

Training feedback: lines matching
    TRAIN_METRICS: epoch=E, loss=L, reg=R, pred=P, probe=Q, time=Ts

Planning feedback: lines matching
    PLAN_METRICS: episode=N, success=True/False, dist=D

Final metric: line matching
    TEST_METRICS: success_rate=X.XX, mean_dist=Y.YYYY, mean_steps_to_success=Z.ZZ

Leaderboard metric keys:
    success_rate_two-rooms, mean_dist_two-rooms, mean_steps_to_success_two-rooms
"""

import re
import sys
from pathlib import Path

# Allow importing from mlsbench package when run standalone
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the jepa-planning (JEPA world model planning) task."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        # Parse training metrics
        train_feedback = self._parse_train_metrics(raw_output)
        if train_feedback:
            feedback_parts.append(train_feedback)

        # Parse planning metrics
        plan_feedback = self._parse_plan_metrics(raw_output)
        if plan_feedback:
            feedback_parts.append(plan_feedback)

        # Parse final test metrics
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
            if "TRAIN_METRICS:" in line:
                lines.append(line.strip())

        if not lines:
            return ""

        summary_lines = lines[-3:]
        return "Training metrics (last epochs):\n" + "\n".join(summary_lines)

    def _parse_plan_metrics(self, output: str) -> str:
        """Extract PLAN_METRICS lines and return a summary."""
        lines = []
        for line in output.splitlines():
            if "PLAN_METRICS:" in line:
                lines.append(line.strip())

        if not lines:
            return ""

        # Show last 5 episode results
        summary_lines = lines[-5:]
        return "Planning episode results (last episodes):\n" + "\n".join(summary_lines)

    def _parse_test_metrics(self, output: str, cmd_label: str) -> tuple:
        """Extract TEST_METRICS line and return feedback + metrics.

        Expected format:
            TEST_METRICS: success_rate=X.XX, mean_dist=Y.YYYY, mean_steps_to_success=Z.ZZ
        """
        metrics: dict = {}
        feedback_parts = []

        for line in output.splitlines():
            if "TEST_METRICS:" not in line:
                continue

            sr_match = re.search(r"success_rate=([\d.]+)", line)
            if sr_match:
                val = float(sr_match.group(1))
                metrics[f"success_rate_{cmd_label}"] = val
                feedback_parts.append(f"Success rate: {val:.2f}")

            dist_match = re.search(r"mean_dist=([\d.]+)", line)
            if dist_match:
                val = float(dist_match.group(1))
                metrics[f"mean_dist_{cmd_label}"] = val
                feedback_parts.append(f"Mean distance: {val:.4f}")

            steps_match = re.search(r"mean_steps_to_success=([\d.]+|nan)", line)
            if steps_match and steps_match.group(1) != "nan":
                val = float(steps_match.group(1))
                metrics[f"mean_steps_to_success_{cmd_label}"] = val
                feedback_parts.append(f"Mean steps to success: {val:.1f}")

        feedback = ", ".join(feedback_parts) if feedback_parts else ""
        return feedback, metrics
