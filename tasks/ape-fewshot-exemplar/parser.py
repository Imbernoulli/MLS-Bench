"""Fail-closed parser for the full official APE evaluation protocol."""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


PROTOCOL = "ape_qwen25_05b_full_official_v1"
EXPECTED = {
    "agnews": {"split": "test", "eval_n": 7_600},
    "sst2": {"split": "validation", "eval_n": 872},
}
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
HEX64 = r"[0-9a-f]{64}"
RESULT_RE = re.compile(
    rf"^APE_RESULT status=complete protocol={PROTOCOL} "
    rf"dataset=(agnews|sst2) eval_split=(test|validation) "
    rf"pool_n=(\d+) dev_n=(\d+) eval_n=(\d+) correct=(\d+) "
    rf"accuracy=({NUMBER}) selected_test_passes=(\d+) "
    rf"n_candidates=(\d+) dev_exec_calls=(\d+) "
    rf"selected_sha256=({HEX64}) data_sha256=({HEX64}) "
    rf"model_sha256=({HEX64}) protocol_sha256=({HEX64}) elapsed=({NUMBER})$"
)
FAILURE_MARKERS = (
    "Traceback (most recent call last)",
    "APE_ORACLE",
    "APE_FAILED",
    "SURFACE_ERROR",
    "EVAL_FAILED",
    "CUDA out of memory",
)


def _reject(reason: str, raw_output: str) -> ParseResult:
    return ParseResult(
        feedback=f"full APE protocol rejected: {reason}\n{raw_output[-3000:]}",
        metrics={},
    )


class Parser(OutputParser):
    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        if cmd_label not in EXPECTED:
            return _reject(f"unknown setting label {cmd_label!r}", raw_output)
        if not raw_output.strip():
            return _reject("empty verifier output", raw_output)
        if any(marker in raw_output for marker in FAILURE_MARKERS):
            return _reject("failure or forbidden test-oracle marker present", raw_output)

        lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        result_lines = [line for line in lines if line.startswith("APE_RESULT")]
        if len(result_lines) != 1 or result_lines[0] != lines[-1]:
            return _reject(
                "completion proof is missing, duplicated, or non-terminal", raw_output
            )
        match = RESULT_RE.fullmatch(result_lines[0])
        if match is None:
            return _reject("completion proof is malformed", raw_output)

        dataset, eval_split = match.group(1, 2)
        try:
            pool_n = int(match.group(3))
            dev_n = int(match.group(4))
            eval_n = int(match.group(5))
            correct = int(match.group(6))
            accuracy = float(match.group(7))
            selected_test_passes = int(match.group(8))
            n_candidates = int(match.group(9))
            dev_exec_calls = int(match.group(10))
            elapsed = float(match.group(15))
        except (TypeError, ValueError, OverflowError):
            return _reject("completion proof contains an invalid scalar", raw_output)

        expected = EXPECTED[cmd_label]
        if dataset != cmd_label or eval_split != expected["split"]:
            return _reject("setting identity or official split does not match", raw_output)
        if pool_n != 128 or dev_n != 200 or eval_n != expected["eval_n"]:
            return _reject("proposal/selection/evaluation inventory is incomplete", raw_output)
        if selected_test_passes != 1:
            return _reject("the selected instruction was not evaluated exactly once", raw_output)
        if not 0 <= correct <= eval_n or not 0.0 <= accuracy <= 1.0:
            return _reject("accuracy or correct-count is outside its valid range", raw_output)
        if not math.isfinite(accuracy) or not math.isclose(
            accuracy, correct / eval_n, rel_tol=0.0, abs_tol=5.1e-10
        ):
            return _reject("accuracy does not match the full correct-count", raw_output)
        if n_candidates < 1 or dev_exec_calls < 0:
            return _reject("candidate or dev-execution proof is invalid", raw_output)
        if not math.isfinite(elapsed) or elapsed <= 0:
            return _reject("runtime proof is invalid", raw_output)

        return ParseResult(
            feedback=(
                f"Completed {dataset} {eval_split}: {correct}/{eval_n} "
                f"({accuracy:.6f}) after train/dev-only candidate ranking; "
                f"selected-test passes=1, elapsed={elapsed:.1f}s."
            ),
            metrics={f"test_acc_{cmd_label}": accuracy},
        )
