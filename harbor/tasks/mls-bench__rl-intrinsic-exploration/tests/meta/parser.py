"""Output parser for rl-intrinsic-exploration."""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parse training and final evaluation metrics for rl-intrinsic-exploration."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics = {}

        train_lines = [
            line.strip()
            for line in raw_output.splitlines()
            if line.strip().startswith("TRAIN_METRICS ")
        ]
        if train_lines:
            feedback_parts.append("Training metrics (last steps):\n" + "\n".join(train_lines[-5:]))

        for line in raw_output.splitlines():
            line = line.strip()
            if not line.startswith("TEST_METRICS "):
                continue
            parsed = {}
            for match in re.finditer(r"(\w+)=([^\s]+)", line):
                key = match.group(1)
                try:
                    val = float(match.group(2))
                except ValueError:
                    continue
                parsed[f"{key}_{cmd_label.replace('-', '_')}"] = val
            if parsed:
                metrics.update(parsed)
                feedback_parts.append("Final evaluation:\n" + line)

        if not feedback_parts:
            feedback_parts.append(raw_output[-3000:])

        return ParseResult(feedback="\n".join(feedback_parts), metrics=metrics)
