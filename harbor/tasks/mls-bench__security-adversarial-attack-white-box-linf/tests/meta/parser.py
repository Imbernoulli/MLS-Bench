"""Task-specific output parser for security-adversarial-attack-white-box-linf.

Expected machine-readable line from run_eval.py:
    ATTACK_METRICS asr=X.XXXX clean_acc=Y.YYYY robust_acc=Z.ZZZZ

Metrics are keyed by command label, e.g.:
    asr_ResNet20_C10
    asr_VGG11BN_C100
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for white-box Linf adversarial attack task."""

    _PATTERN = re.compile(
        r"ATTACK_METRICS\s+"
        r"asr=([\d.eE+\-]+)\s+"
        r"clean_acc=([\d.eE+\-]+)\s+"
        r"robust_acc=([\d.eE+\-]+)"
    )

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        metrics: dict = {}
        feedback_parts: list[str] = []

        for line in raw_output.splitlines():
            m = self._PATTERN.search(line)
            if not m:
                continue

            asr = float(m.group(1))
            clean_acc = float(m.group(2))
            robust_acc = float(m.group(3))
            suffix = cmd_label.replace("-", "_")

            metrics[f"asr_{suffix}"] = asr
            feedback_parts.append(
                f"Attack results ({cmd_label}): "
                f"ASR={asr:.4f}, clean_acc={clean_acc:.4f}, robust_acc={robust_acc:.4f}"
            )

        feedback = "\n".join(feedback_parts) if feedback_parts else raw_output
        return ParseResult(feedback=feedback, metrics=metrics)
