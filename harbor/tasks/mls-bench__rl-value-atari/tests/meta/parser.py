"""Task-specific output parser for rl-value-atari.

Handles combined train+eval output from CleanRL value-based algorithms:

Training feedback: lines matching
    TRAIN_METRICS step=N key=val key=val ...

Evaluation feedback: lines matching
    Eval episodic_return: X.XX

Metrics are keyed by environment name, e.g. eval_return_breakoutnoframeskip_v4.
"""

import re
import sys
from pathlib import Path

# Allow importing from mlsbench package when run standalone
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the rl-value-atari task."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        # Parse training metrics
        train_feedback = self._parse_train_metrics(raw_output)
        if train_feedback:
            feedback_parts.append(train_feedback)

        # Parse evaluation returns
        eval_feedback, eval_metrics = self._parse_eval_returns(raw_output, cmd_label)
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
        return "Training metrics (last steps):\n" + "\n".join(summary_lines)

    def _parse_eval_returns(self, output: str, cmd_label: str) -> tuple[str, dict]:
        """Extract Eval episodic_return lines and return feedback + metrics.

        Expected format: Eval episodic_return: X.XX
        """
        returns: list[float] = []
        eval_lines: list[str] = []

        for line in output.splitlines():
            match = re.search(
                r"Eval episodic_return:\s*(-?[\d.]+(?:e[+-]?\d+)?|nan|inf|-inf)",
                line, re.IGNORECASE,
            )
            if match:
                raw = match.group(1).lower()
                val = float(raw)
                eval_lines.append(line.strip())
                if not (val != val or abs(val) == float("inf")):
                    returns.append(val)

        metrics: dict = {}
        feedback = ""

        if returns:
            final_return = returns[-1]
            metric_key = "eval_return_" + cmd_label.replace("-", "_")
            metrics[metric_key] = final_return

            feedback = f"Evaluation ({cmd_label}):\n" + "\n".join(eval_lines[-3:])
            feedback += f"\nFinal eval return: {final_return:.2f}"

        return feedback, metrics
