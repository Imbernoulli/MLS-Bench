"""Output parser for the optimization-parity task."""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


_PAIR_RE = re.compile(r"(\w+)=([\d.eE+-]+)")


class Parser(OutputParser):
    """Parse TRAIN_METRICS / RUN_METRICS / FINAL_METRICS output."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts: list[str] = []
        metrics: dict[str, float] = {}

        train_lines = [
            line.strip()
            for line in raw_output.splitlines()
            if line.strip().startswith("TRAIN_METRICS")
        ]
        if train_lines:
            feedback_parts.append("Training progress:\n" + "\n".join(train_lines[-5:]))

        run_lines = [
            line.strip()
            for line in raw_output.splitlines()
            if line.strip().startswith("RUN_METRICS")
        ]
        if run_lines:
            feedback_parts.append("Recent runs:\n" + "\n".join(run_lines[-5:]))

        final_lines = [
            line.strip()
            for line in raw_output.splitlines()
            if line.strip().startswith("FINAL_METRICS")
        ]
        if final_lines:
            final_line = final_lines[-1]
            for key, value in _PAIR_RE.findall(final_line):
                metric_key = key if cmd_label == "eval" else f"{key}_{cmd_label}"
                metrics[metric_key] = float(value)
            feedback_parts.append("Final metrics:\n" + final_line)

        if not feedback_parts:
            feedback_parts.append(raw_output[-3000:])

        return ParseResult(feedback="\n\n".join(feedback_parts), metrics=metrics)
