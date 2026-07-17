"""Task-specific output parser for rl-offpolicy-sample-efficiency.

Training feedback: lines matching
    TRAIN_METRICS step=S eval_return=R

Final evaluation: lines matching
    TEST_METRICS mean_reward=M std_reward=S

Leaderboard metrics: mean_reward (higher is better).
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the rl-offpolicy-sample-efficiency task."""

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
            if "TRAIN_METRICS " in line:
                lines.append(line.strip())

        if not lines:
            return ""

        summary_lines = lines[-5:]
        return "Training metrics (last evaluations):\n" + "\n".join(summary_lines)

    def _parse_test_metrics(self, output: str, cmd_label: str) -> tuple[str, dict]:
        """Extract TEST_METRICS lines.

        Expected format: TEST_METRICS mean_reward=X.XXXX std_reward=X.XXXX
        """
        mean_rewards: list[float] = []
        std_rewards: list[float] = []
        test_lines: list[str] = []

        for line in output.splitlines():
            match = re.search(
                r"TEST_METRICS\s+mean_reward=([-\d.]+)\s+std_reward=([-\d.]+)", line
            )
            if match:
                test_lines.append(line.strip())
                mean_rewards.append(float(match.group(1)))
                std_rewards.append(float(match.group(2)))

        metrics: dict = {}
        feedback = ""

        if mean_rewards:
            final_mean = mean_rewards[-1]
            final_std = std_rewards[-1]
            suffix = cmd_label.replace("-", "_")
            metrics[f"mean_reward_{suffix}"] = final_mean
            metrics[f"std_reward_{suffix}"] = final_std

            feedback = "Final evaluation:\n" + "\n".join(test_lines[-3:])
            feedback += f"\nMean reward: {final_mean:.4f} +/- {final_std:.4f}"

        return feedback, metrics
