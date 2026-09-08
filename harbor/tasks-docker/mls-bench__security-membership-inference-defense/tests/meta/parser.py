"""Parser for security-membership-inference-defense."""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Extract membership-inference metrics from TEST_METRICS lines."""

    _TEST_PATTERN = re.compile(
        r"TEST_METRICS\s+"
        r"test_acc=([\d.eE+\-]+)\s+"
        r"mia_auc=([\d.eE+\-]+)\s+"
        r"privacy_gap=([\d.eE+\-]+)\s+"
        r"privacy_score=([\d.eE+\-]+)"
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
            test_acc, mia_auc, privacy_gap, privacy_score = map(float, match.groups())
            metrics[f"test_acc_{suffix}"] = test_acc
            metrics[f"mia_auc_{suffix}"] = mia_auc
            metrics[f"privacy_gap_{suffix}"] = privacy_gap
            metrics[f"privacy_score_{suffix}"] = privacy_score
            feedback_parts.append(
                f"{cmd_label}: test_acc={test_acc:.4f}, mia_auc={mia_auc:.4f}, "
                f"privacy_gap={privacy_gap:.4f}, privacy_score={privacy_score:.4f}"
            )

        feedback = "\n".join(feedback_parts) if feedback_parts else raw_output[-3000:]
        return ParseResult(feedback=feedback, metrics=metrics)
