"""Parser for the llm-kv-selection-budgeting task."""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Extracts final benchmark metrics."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        final_feedback, final_metrics = self._parse_test_metrics(raw_output, cmd_label)
        if final_feedback:
            feedback_parts.append(final_feedback)
        metrics.update(final_metrics)

        feedback = "\n".join(feedback_parts) if feedback_parts else raw_output
        return ParseResult(feedback=feedback, metrics=metrics)

    def _parse_test_metrics(self, output: str, cmd_label: str) -> tuple[str, dict]:
        metrics: dict = {}
        feedback = ""

        for line in output.splitlines():
            if "TEST_METRICS:" not in line:
                continue
            pairs = dict(re.findall(r"(\w+)=([-+]?[\d.]+(?:e[+-]?\d+)?)", line, re.IGNORECASE))
            if "final_score" in pairs:
                metrics[f"final_score_{cmd_label}"] = float(pairs["final_score"])
            if "mean_retained_fraction" in pairs:
                metrics[f"mean_retained_fraction_{cmd_label}"] = float(pairs["mean_retained_fraction"])
            if "runtime_seconds" in pairs:
                metrics[f"runtime_seconds_{cmd_label}"] = float(pairs["runtime_seconds"])
            if metrics:
                parts = [f"{key}={value:.4f}" for key, value in sorted(metrics.items())]
                feedback = f"Final metrics ({cmd_label}): " + ", ".join(parts)

        return feedback, metrics
