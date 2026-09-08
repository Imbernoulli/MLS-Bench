"""Task-specific output parser for tdmpc2-planning.

Handles output from TD-MPC2 online training:

Training feedback: lines matching
    TRAIN_METRICS step=N episode_reward=X.XX

Evaluation feedback: lines matching
    EVAL_METRIC step=N episode_reward=X.XX

Metrics are keyed by environment name from cmd_label.
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for tdmpc2-planning task."""

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
        return "Training metrics (last episodes):\n" + "\n".join(summary_lines)

    def _parse_eval_metrics(
        self, output: str, cmd_label: str
    ) -> tuple[str, dict]:
        """Extract EVAL_METRIC lines and return feedback + metrics.

        Expected format: EVAL_METRIC step=N episode_reward=X.XX
        """
        eval_values: list[tuple[int, float]] = []
        eval_lines: list[str] = []

        for line in output.splitlines():
            if not line.strip().startswith("EVAL_METRIC "):
                continue
            eval_lines.append(line.strip())
            step_match = re.search(r"step=(\d+)", line)
            reward_match = re.search(
                r"episode_reward=(-?[\d.]+(?:e[+-]?\d+)?)", line
            )
            if step_match and reward_match:
                step = int(step_match.group(1))
                reward = float(reward_match.group(1))
                eval_values.append((step, reward))

        metrics: dict = {}
        feedback = ""

        if eval_values:
            final_step, final_reward = eval_values[-1]
            metric_key = "episode_reward_" + cmd_label.replace("-", "_")
            metrics[metric_key] = final_reward

            feedback = f"Evaluation ({cmd_label}):\n"
            feedback += "\n".join(eval_lines[-3:])
            feedback += f"\nFinal eval reward: {final_reward:.2f} (step {final_step})"

        return feedback, metrics
