"""Parser for dlm-dkv-policy real-rollout evaluation."""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Extracts real-rollout benchmark metrics from LLaDA evaluation output."""

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
            for key, raw in re.findall(
                r"([A-Za-z_][A-Za-z0-9_]*)=([-+]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
                line,
                flags=re.IGNORECASE,
            ):
                metrics[f"{key}_{cmd_label}"] = float(raw)
            if f"final_score_{cmd_label}" in metrics:
                feedback = f"Final score ({cmd_label}): {metrics[f'final_score_{cmd_label}']:.4f}"

        return feedback, metrics
