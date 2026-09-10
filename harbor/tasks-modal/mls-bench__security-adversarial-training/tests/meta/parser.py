"""Output parser for security-adversarial-training task.

Expected machine-readable line from run_adv_train.py:
    TEST_METRICS clean_acc=X.XXXX robust_acc_fgsm=Y.YYYY robust_acc_pgd=Z.ZZZZ

Metrics keyed by command label, e.g.:
    robust_acc_pgd_SmallCNN_MNIST
    robust_acc_pgd_PreActResNet18_C10
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for adversarial training robustness task."""

    _TRAIN_PATTERN = re.compile(
        r"TRAIN_METRICS\s+epoch=(\d+)\s+loss=([\d.eE+\-]+)"
    )
    _TEST_PATTERN = re.compile(
        r"TEST_METRICS\s+"
        r"clean_acc=([\d.eE+\-]+)\s+"
        r"robust_acc_fgsm=([\d.eE+\-]+)\s+"
        r"robust_acc_pgd=([\d.eE+\-]+)"
    )

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        metrics: dict = {}
        feedback_parts: list[str] = []

        # Training progress
        train_lines = []
        for line in raw_output.splitlines():
            if self._TRAIN_PATTERN.search(line):
                train_lines.append(line.strip())
        if train_lines:
            feedback_parts.append(
                f"Training progress ({cmd_label}):\n" +
                "\n".join(train_lines[-5:])
            )

        # Test metrics
        for line in raw_output.splitlines():
            m = self._TEST_PATTERN.search(line)
            if not m:
                continue

            clean_acc = float(m.group(1))
            robust_fgsm = float(m.group(2))
            robust_pgd = float(m.group(3))
            suffix = cmd_label.replace("-", "_")

            metrics[f"clean_acc_{suffix}"] = clean_acc
            metrics[f"robust_acc_fgsm_{suffix}"] = robust_fgsm
            metrics[f"robust_acc_pgd_{suffix}"] = robust_pgd

            feedback_parts.append(
                f"Results ({cmd_label}): "
                f"clean_acc={clean_acc:.4f}, "
                f"robust_acc_fgsm={robust_fgsm:.4f}, "
                f"robust_acc_pgd={robust_pgd:.4f}"
            )

        feedback = "\n".join(feedback_parts) if feedback_parts else raw_output[-3000:]
        return ParseResult(feedback=feedback, metrics=metrics)
