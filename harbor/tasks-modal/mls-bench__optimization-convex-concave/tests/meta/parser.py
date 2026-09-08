"""Output parser for the optimization-convex-concave task."""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


_PAIR_RE = re.compile(r"([A-Za-z0-9_]+)=([-+0-9.eE]+)")


class Parser(OutputParser):
    """Parse STEP_METRICS / RUN_METRICS / FINAL_METRICS output."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts: list[str] = []
        metrics: dict[str, float] = {}

        step_lines = [
            line.strip()
            for line in raw_output.splitlines()
            if line.strip().startswith("STEP_METRICS")
        ]
        if step_lines:
            feedback_parts.append("Recent checkpoints:\n" + "\n".join(step_lines[-6:]))

        run_lines = [
            line.strip()
            for line in raw_output.splitlines()
            if line.strip().startswith("RUN_METRICS")
        ]
        if run_lines:
            feedback_parts.append("Recent runs:\n" + "\n".join(run_lines[-6:]))

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
