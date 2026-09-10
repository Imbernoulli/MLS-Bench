"""Task-specific output parser for agent-tool-reasoning.

Dispatches on cmd_label (3 settings). Each setting emits
    TEST_METRICS: pass_rate=X avg_queries=X give_up_rate=X answer_ts=<ts>
from train.sh. Metric names are suffixed per-backend so a single
leaderboard row carries results across all 3 settings:

  I1-instruction-deepseek -> pass_rate_deepseek, avg_queries_deepseek,
                             give_up_rate_deepseek, answer_ts_deepseek
  I1-instruction-qwen72b  -> ..._qwen72b
  I1-instruction-qwen7b   -> ..._qwen7b

answer_ts is the UTC start timestamp of the specific test invocation
(unique across rounds in the same agent run). It lets compute_sopr.sh
locate the exact answer-file directory that produced these metrics
for post-hoc SoPR judging.

SoPR is computed post-hoc by scripts/compute_sopr.sh and written
directly into leaderboard.csv, not extracted here.
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


_SUFFIX_MAP = {
    "I1-instruction-deepseek": "_deepseek",
    "I1-instruction-qwen72b": "_qwen72b",
    "I1-instruction-qwen7b": "_qwen7b",
}


class Parser(OutputParser):
    """Parser for the agent-tool-reasoning (StableToolBench) task."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        suffix = _SUFFIX_MAP.get(cmd_label, "")

        feedback_parts: list[str] = []
        metrics: dict = {}

        train_feedback = self._parse_train_metrics(raw_output)
        if train_feedback:
            feedback_parts.append(train_feedback)

        test_feedback, test_metrics = self._parse_test_metrics(raw_output, suffix)
        if test_feedback:
            feedback_parts.append(test_feedback)
        metrics.update(test_metrics)

        feedback = "\n".join(feedback_parts) if feedback_parts else raw_output
        return ParseResult(feedback=feedback, metrics=metrics)

    def _parse_train_metrics(self, output: str) -> str:
        lines = [
            l.strip() for l in output.splitlines()
            if l.strip().startswith("TRAIN_METRICS:")
        ]
        if not lines:
            return ""
        return "Training metrics:\n" + "\n".join(lines[-3:])

    def _parse_test_metrics(self, output: str, suffix: str) -> tuple[str, dict]:
        metrics: dict = {}
        feedback = ""

        for line in output.splitlines():
            match = re.search(
                r"TEST_METRICS:\s+pass_rate=([\d.]+)\s+avg_queries=([\d.]+)\s+give_up_rate=([\d.]+)(?:\s+answer_ts=(\S+))?",
                line,
            )
            if match:
                pass_rate = float(match.group(1))
                avg_queries = float(match.group(2))
                give_up_rate = float(match.group(3))
                answer_ts = match.group(4) or ""
                metrics[f"pass_rate{suffix}"] = pass_rate
                metrics[f"avg_queries{suffix}"] = avg_queries
                metrics[f"give_up_rate{suffix}"] = give_up_rate
                if answer_ts:
                    metrics[f"answer_ts{suffix}"] = answer_ts
                feedback = (
                    f"Test evaluation ({suffix.lstrip('_') or 'default'}):\n"
                    f"  Pass rate: {pass_rate:.4f}\n"
                    f"  Avg queries: {avg_queries:.1f}\n"
                    f"  Give-up rate: {give_up_rate:.4f}"
                )

        return feedback, metrics
