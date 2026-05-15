"""Task-specific output parser for rl-offline-off2on.

Handles combined train+eval output from CORL offline-to-online algorithms:

Training feedback: lines matching
    TRAIN_METRICS step=N key=val key=val ...

Evaluation feedback: lines matching
    D4RL score: X.XXX

Metrics are keyed by dataset name, e.g. d4rl_score_pen_cloned_v1.
"""

import re
import sys
from pathlib import Path

# Allow importing from mlsbench package when run standalone
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the rl-offline-off2on task."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        # Parse training metrics
        train_feedback = self._parse_train_metrics(raw_output)
        if train_feedback:
            feedback_parts.append(train_feedback)

        # Parse D4RL evaluation scores
        eval_feedback, eval_metrics = self._parse_eval_scores(raw_output, cmd_label)
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

        # Return last 5 training metric lines as feedback
        summary_lines = lines[-5:]
        return "Training metrics (last steps):\n" + "\n".join(summary_lines)

    def _parse_eval_scores(self, output: str, cmd_label: str) -> tuple[str, dict]:
        """Extract D4RL score lines and return feedback + metrics.

        Expected format: D4RL score: X.XXX
        The cmd_label identifies the dataset (e.g. 'pen-cloned-v1').
        """
        scores: list[float] = []
        eval_lines: list[str] = []

        for line in output.splitlines():
            match = re.search(r"D4RL score:\s*(-?[\d.]+(?:e[+-]?\d+)?|nan|inf|-inf)", line, re.IGNORECASE)
            if match:
                raw = match.group(1).lower()
                score = float(raw)
                eval_lines.append(line.strip())
                if not (score != score or abs(score) == float("inf")):
                    scores.append(score)

        metrics: dict = {}
        feedback = ""

        if scores:
            final_score = scores[-1]
            metric_key = "d4rl_score_" + cmd_label.replace("-", "_")
            metrics[metric_key] = final_score

            feedback = f"D4RL evaluation ({cmd_label}):\n" + "\n".join(eval_lines[-3:])
            feedback += f"\nFinal D4RL score: {final_score:.3f}"

        return feedback, metrics
