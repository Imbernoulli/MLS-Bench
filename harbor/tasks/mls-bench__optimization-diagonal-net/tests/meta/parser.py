"""Output parser for the opt-diagonal-net task."""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


_PAIR_RE = re.compile(r"([A-Za-z0-9_]+)=([-+0-9.eE]+)")


class Parser(OutputParser):
    """Parse SEARCH_METRICS / FINAL_METRICS output from opt-diagonal-net."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts: list[str] = []
        metrics: dict[str, float] = {}

        # --- SEARCH_METRICS (feedback only) ---
        search_lines = [
            line.strip()
            for line in raw_output.splitlines()
            if line.strip().startswith("SEARCH_METRICS")
        ]
        if search_lines:
            feedback_parts.append(
                "Search progress:\n" + "\n".join(search_lines[-8:])
            )

        # --- FINAL_METRICS (feedback + leaderboard) ---
        final_lines = [
            line.strip()
            for line in raw_output.splitlines()
            if line.strip().startswith("FINAL_METRICS")
        ]
        if final_lines:
            final_line = final_lines[-1]
            for key, value in _PAIR_RE.findall(final_line):
                metric_key = f"{key}_{cmd_label}"
                metrics[metric_key] = float(value)
            feedback_parts.append("Final metrics:\n" + final_line)

        # --- Fallback ---
        if not feedback_parts:
            feedback_parts.append(raw_output[-3000:])

        return ParseResult(
            feedback="\n\n".join(feedback_parts),
            metrics=metrics,
        )
