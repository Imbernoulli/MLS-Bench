"""Parser for security-poison-robust-learning."""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Extract poison-robust-learning metrics from TEST_METRICS lines."""

    _TEST_PATTERN = re.compile(
        r"TEST_METRICS\s+"
        r"test_acc=([\d.eE+\-]+)\s+"
        r"poison_fit=([\d.eE+\-]+)\s+"
        r"robust_score=([\d.eE+\-]+)"
    )

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        suffix = cmd_label.replace("-", "_")
        metrics = {}
        feedback_parts = []

        for line in raw_output.splitlines():
            line = line.strip()
            if line.startswith("TRAIN_METRICS"):
                feedback_parts.append(line)
                continue
            match = self._TEST_PATTERN.search(line)
            if not match:
                continue
            test_acc, poison_fit, robust_score = map(float, match.groups())
            metrics[f"test_acc_{suffix}"] = test_acc
            metrics[f"poison_fit_{suffix}"] = poison_fit
            metrics[f"robust_score_{suffix}"] = robust_score
            feedback_parts.append(
                f"{cmd_label}: test_acc={test_acc:.4f}, poison_fit={poison_fit:.4f}, "
                f"robust_score={robust_score:.4f}"
            )

        feedback = "\n".join(feedback_parts) if feedback_parts else raw_output[-3000:]
        return ParseResult(feedback=feedback, metrics=metrics)
