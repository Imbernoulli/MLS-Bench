"""Task-specific output parser for pla-binding-affinity.
Handles output from custom_pla.py:
- Training feedback: TRAIN_METRICS epoch=N loss=val val_rmse=val val_rp=val
- Test feedback: TEST_METRICS rmse=val rp=val
Metrics are keyed by benchmark label, e.g. rmse_PDBbind2013, rp_PDBbind2016.
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Only reach for the repo checkout when mlsbench is genuinely absent.
# Under Harbor this parser is exec'd from /tmp/mlsbench-verifier.XXXXXX/meta/,
# so a __file__-derived root is /tmp and this would put an agent-writable
# directory at the front of the verifier's sys.path.
try:  # noqa: E402
    import mlsbench  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - native checkout, no install
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


def _failure_excerpt(raw_output: str, max_lines: int = 30, max_chars: int = 2000) -> str:
    """Preserve crash evidence when no final metrics were produced.

    This parser replaces the raw output with its own summary whenever any
    progress line was seen.  A run that printed progress and then died -- a CUDA
    OOM stolen by a co-tenant, an OOM-killed process, an exception in eval --
    therefore surfaced as a bare rc=1 with no traceback anywhere, which is
    indistinguishable from a code fault and is how bad rows reach a leaderboard.
    When the final metrics are absent, append the tail of the real output
    (preferring the traceback) so the failure stays visible.
    """
    if not raw_output:
        return ""
    lines = raw_output.splitlines()
    start = max(0, len(lines) - max_lines)
    for i, ln in enumerate(lines):
        if "Traceback (most recent call last)" in ln:
            start = i
            break
    tail = "\n".join(lines[start:])[-max_chars:]
    if not tail.strip():
        return ""
    return "No final metrics were produced; raw output tail:\n" + tail


class Parser(OutputParser):
    """Parser for the pla-binding-affinity task."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        train_feedback = self._parse_train_metrics(raw_output)
        if train_feedback:
            feedback_parts.append(train_feedback)

        eval_feedback, eval_metrics = self._parse_eval_metrics(raw_output, cmd_label)
        if eval_feedback:
            feedback_parts.append(eval_feedback)
        metrics.update(eval_metrics)

        if not metrics:
            _tail = _failure_excerpt(raw_output)
            if _tail:
                feedback_parts.append(_tail)
        feedback = "\n".join(feedback_parts) if feedback_parts else raw_output
        return ParseResult(feedback=feedback, metrics=metrics)

    def _parse_train_metrics(self, output: str) -> str:
        lines = [l.strip() for l in output.splitlines() if l.strip().startswith("TRAIN_METRICS ")]
        if not lines:
            return ""
        return "Training progress (last 5 reports):\n" + "\n".join(lines[-5:])

    def _parse_eval_metrics(self, output: str, cmd_label: str) -> tuple[str, dict]:
        metrics: dict = {}
        feedback_parts = []

        for line in output.splitlines():
            line = line.strip()
            if not line.startswith("TEST_METRICS "):
                continue
            parts = line[len("TEST_METRICS "):].strip()
            # Match metric=value pairs
            for match in re.finditer(r"(\w+)=([\d.eE+-]+)", parts):
                metric_name = match.group(1).strip()
                value = float(match.group(2))
                key = f"{metric_name}_{cmd_label}"
                metrics[key] = value
                feedback_parts.append(f"  {metric_name}: {value:.6f}")

        feedback = ""
        if feedback_parts:
            feedback = f"Test results ({cmd_label}):\n" + "\n".join(feedback_parts)

        return feedback, metrics
