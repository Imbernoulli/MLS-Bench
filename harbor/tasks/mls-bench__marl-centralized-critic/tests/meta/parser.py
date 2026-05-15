"""Task-specific output parser for marl-centralized-critic.

Training feedback: lines matching
    TRAIN_METRICS t_env=T return_mean=R return_std=S battle_won_mean=W

Test feedback: lines matching
    TEST_METRICS t_env=T return_mean=R return_std=S battle_won_mean=W

The battle_won_mean field is the SMAC win rate (primary metric).
Leaderboard metrics are keyed by map label so multiple maps can coexist
in the same final row, e.g. test_battle_won_mean_2s_vs_1sc.
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


_NUM = r"[\d.eE+-]+"
_TEST_RE = re.compile(
    r"TEST_METRICS\s+t_env=(\d+)\s+"
    rf"return_mean=({_NUM})\s+"
    rf"return_std=({_NUM})\s+"
    rf"battle_won_mean=({_NUM})"
)


class Parser(OutputParser):
    """Parser for the marl-centralized-critic task."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts: list[str] = []
        metrics: dict = {}

        train_feedback = self._parse_train_metrics(raw_output)
        if train_feedback:
            feedback_parts.append(train_feedback)

        test_feedback, test_metrics = self._parse_test_metrics(raw_output, cmd_label)
        if test_feedback:
            feedback_parts.append(test_feedback)
        metrics.update(test_metrics)

        if feedback_parts:
            feedback = "\n".join(feedback_parts)
        else:
            feedback = raw_output

        return ParseResult(feedback=feedback, metrics=metrics)

    def _parse_train_metrics(self, output: str) -> str:
        lines = [ln.strip() for ln in output.splitlines() if ln.strip().startswith("TRAIN_METRICS ")]
        if not lines:
            return ""
        return "Training metrics (last steps):\n" + "\n".join(lines[-5:])

    def _parse_test_metrics(self, output: str, cmd_label: str) -> tuple[str, dict]:
        metrics: dict = {}
        feedback = ""
        label_key = cmd_label.replace("-", "_")
        for line in output.splitlines():
            m = _TEST_RE.search(line)
            if not m:
                continue
            t_env = int(m.group(1))
            return_mean = float(m.group(2))
            return_std = float(m.group(3))
            battle_won_mean = float(m.group(4))
            metrics[f"test_return_mean_{label_key}"] = return_mean
            metrics[f"test_return_std_{label_key}"] = return_std
            metrics[f"test_battle_won_mean_{label_key}"] = battle_won_mean
            feedback = (
                f"Final test for {cmd_label} (t_env={t_env}):\n"
                f"  Win rate: {battle_won_mean:.4f}\n"
                f"  Mean return: {return_mean:.4f} +/- {return_std:.4f}"
            )
        return feedback, metrics
