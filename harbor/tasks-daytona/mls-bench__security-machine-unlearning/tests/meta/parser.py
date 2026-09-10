"""Parser for security-machine-unlearning."""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Extract unlearning metrics from TEST_METRICS lines."""

    _TEST_PATTERN = re.compile(
        r"TEST_METRICS\s+"
        r"retain_acc=([\d.eE+\-]+)\s+"
        r"forget_acc=([\d.eE+\-]+)\s+"
        r"forget_mia_auc=([\d.eE+\-]+)\s+"
        r"unlearn_score=([\d.eE+\-]+)"
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
            retain_acc, forget_acc, forget_mia_auc, unlearn_score = map(float, match.groups())
            metrics[f"retain_acc_{suffix}"] = retain_acc
            metrics[f"forget_acc_{suffix}"] = forget_acc
            metrics[f"forget_mia_auc_{suffix}"] = forget_mia_auc
            metrics[f"unlearn_score_{suffix}"] = unlearn_score
            feedback_parts.append(
                f"{cmd_label}: retain_acc={retain_acc:.4f}, forget_acc={forget_acc:.4f}, "
                f"forget_mia_auc={forget_mia_auc:.4f}, unlearn_score={unlearn_score:.4f}"
            )

        feedback = "\n".join(feedback_parts) if feedback_parts else raw_output[-3000:]
        return ParseResult(feedback=feedback, metrics=metrics)
