"""Task-specific output parser for llm-offline-rl.

Training feedback: trainer metric dict lines from train.sh, e.g.
  {'loss': 0.6857, 'rewards/chosen': 0.0029, ...}

Math eval feedback: lines emitted by scripts/math_eval.py, one per benchmark:
  EVAL_RESULT benchmark=<name> correct=<int> total=<int> accuracy=<pct>

Leaderboard metrics:
  gsm8k_accuracy, math500_accuracy, aime2024_accuracy   (math_eval)
"""

import ast
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


# benchmark name in EVAL_RESULT line → leaderboard metric key
_METRIC_MAP = {
    "gsm8k": "gsm8k_accuracy",
    "math500": "math500_accuracy",
    "aime2024": "aime2024_accuracy",
}


class Parser(OutputParser):
    """Parser for the llm-offline-rl task."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        if cmd_label == "train":
            return self._parse_train(raw_output)
        if cmd_label == "math_eval":
            return self._parse_math_eval(raw_output)
        return super().parse(cmd_label, raw_output)

    # ------------------------------------------------------------------
    # Training output: trainer metric dict lines
    # ------------------------------------------------------------------

    def _parse_train(self, output: str) -> ParseResult:
        metric_lines: list[str] = []
        last_metrics: dict = {}

        for line in output.splitlines():
            stripped = line.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    d = ast.literal_eval(stripped)
                except (ValueError, SyntaxError):
                    continue
                if isinstance(d, dict):
                    metric_lines.append(stripped)
                    last_metrics = {k: _try_float(v) for k, v in d.items()}

        feedback = "\n".join(metric_lines) if metric_lines else output
        return ParseResult(feedback=feedback, metrics=last_metrics)

    # ------------------------------------------------------------------
    # Math eval output: EVAL_RESULT lines from math_eval.py
    # ------------------------------------------------------------------

    def _parse_math_eval(self, output: str) -> ParseResult:
        metrics: dict = {}
        feedback_lines: list[str] = []

        pattern = re.compile(
            r"EVAL_RESULT\s+benchmark=(\S+)\s+correct=(\d+)\s+total=(\d+)\s+accuracy=([\d.]+)"
        )
        for line in output.splitlines():
            m = pattern.search(line)
            if not m:
                continue
            name = m.group(1)
            correct = int(m.group(2))
            total = int(m.group(3))
            accuracy = float(m.group(4))
            key = _METRIC_MAP.get(name)
            if key:
                metrics[key] = accuracy
            feedback_lines.append(
                f"{name}: {correct}/{total} = {accuracy:.2f}%"
            )

        feedback = "\n".join(feedback_lines) if feedback_lines else output
        return ParseResult(feedback=feedback, metrics=metrics)


def _try_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return v
