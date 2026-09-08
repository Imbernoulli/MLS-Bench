"""Parser for the llm-kv-adaptive-quantization task."""

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

        test_feedback, test_metrics = self._parse_test_metrics(raw_output, cmd_label)
        if test_feedback:
            feedback_parts.append(test_feedback)
        metrics.update(test_metrics)

        feedback = "\n".join(feedback_parts) if feedback_parts else raw_output
        return ParseResult(feedback=feedback, metrics=metrics)

    def _parse_test_metrics(self, output: str, cmd_label: str) -> tuple[str, dict]:
        metrics: dict = {}

        for line in output.splitlines():
            if "TEST_METRICS:" not in line:
                continue
            for key, value in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([-+]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", line, re.IGNORECASE):
                metrics[f"{key}_{cmd_label}"] = float(value)

        if not metrics:
            return "", metrics
        summary = ", ".join(
            f"{name.removesuffix('_' + cmd_label)}={value:.4f}"
            for name, value in sorted(metrics.items())
            if name.endswith(f"_{cmd_label}")
        )
        return f"Metrics ({cmd_label}): {summary}", metrics
