"""Strict output parser for full-split constrained choice decoding.

Harness emits one metric line per run:
    CD_METRICS valid_rate=<V> accuracy=<A> n=<N> elapsed=<T>

Leaderboard metrics (higher is better):
    accuracy_{label}     (# valid AND correct) / n   -- PRIMARY (un-gameable)
    valid_rate_{label}   # structurally valid / n    -- reported for diagnosis
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


_PROTOCOL = "constrained-decoding-full-v3"
_EXPECTED_TASK = "cd-choice-verbalizer"
_EXPECTED_SURFACE = "decoder_choice_verbalizer"
_EXPECTED_LABEL = "agnews"
_EXPECTED_N = 7600


_FAILURE_MARKER = re.compile(
    r"^(?:CD_(?:FAILED|FAILURE)\b|Traceback \(most recent call last\):|"
    r"(?:[A-Za-z_][A-Za-z0-9_]*Error|ERROR|Exception)(?::|\b))",
    re.IGNORECASE,
)


class Parser(OutputParser):
    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        expected_label = _EXPECTED_LABEL
        expected_n = _EXPECTED_N
        if cmd_label != expected_label:
            return ParseResult(feedback="Rejected unexpected evaluation label", metrics={})
        identity = (
            rf"protocol={re.escape(_PROTOCOL)} "
            rf"task={re.escape(_EXPECTED_TASK)} "
            rf"surface={re.escape(_EXPECTED_SURFACE)} "
        )
        pattern = re.compile(
            rf"CD_METRICS {identity}dataset={re.escape(expected_label)} "
            r"valid_rate=([\d.eE+-]+)\s+"
            r"accuracy=([\d.eE+-]+)\s+n=(\d+)\s+elapsed=([\d.eE+-]+)"
        )
        lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        if any(_FAILURE_MARKER.match(line) for line in lines):
            return ParseResult(
                feedback="Rejected explicit verification failure marker",
                metrics={},
            )

        model_prefix_lines = [line for line in lines if line.startswith("CD_MODEL")]
        data_prefix_lines = [line for line in lines if line.startswith("CD_DATA")]
        metric_lines = [line for line in lines if line.startswith("CD_METRICS")]
        complete_prefix_lines = [line for line in lines if line.startswith("CD_COMPLETE")]
        if len(metric_lines) != 1:
            return ParseResult(
                feedback=f"Rejected verification: expected one CD_METRICS line, got {len(metric_lines)}",
                metrics={},
            )
        match = pattern.fullmatch(metric_lines[0])
        if match is None:
            return ParseResult(feedback="Rejected malformed CD_METRICS line", metrics={})
        vr = float(match.group(1))
        acc = float(match.group(2))
        n = int(match.group(3))
        elapsed = float(match.group(4))
        expected_model = re.compile(
            rf"CD_MODEL {identity}params=494032768 "
            r"device=cuda:\d+ dtype=torch\.float16"
        )
        expected_data = (
            f"CD_DATA protocol={_PROTOCOL} task={_EXPECTED_TASK} "
            f"surface={_EXPECTED_SURFACE} dataset={expected_label} "
            f"n={expected_n} seed=42"
        )
        expected_complete = (
            f"CD_COMPLETE protocol={_PROTOCOL} task={_EXPECTED_TASK} "
            f"surface={_EXPECTED_SURFACE} dataset={expected_label} "
            f"n={expected_n} seed=42 status=ok"
        )
        proof_lines = (
            len(model_prefix_lines) == 1
            and expected_model.fullmatch(model_prefix_lines[0]) is not None
            and data_prefix_lines == [expected_data]
            and complete_prefix_lines == [expected_complete]
        )
        proof_order = proof_lines and (
            lines.index(model_prefix_lines[0])
            < lines.index(data_prefix_lines[0])
            < lines.index(metric_lines[0])
            < lines.index(complete_prefix_lines[0])
        )
        completion_is_final = bool(lines) and lines[-1] == expected_complete
        if (not all(math.isfinite(value) for value in (vr, acc, elapsed))
                or not 0.0 <= vr <= 1.0 or not 0.0 <= acc <= 1.0
                or elapsed <= 0.0 or n != expected_n
                or not proof_order or not completion_is_final):
            return ParseResult(
                feedback=(
                    f"Rejected incomplete verification proof: n={n}, "
                    f"expected {expected_n}"
                ),
                metrics={},
            )
        metrics = {
            f"accuracy_{cmd_label}": acc,
            f"valid_rate_{cmd_label}": vr,
        }
        feedback = (
            f"Results ({cmd_label}):\n"
            f"  accuracy:    {acc:.6f}   (valid AND correct / n; higher is better)\n"
            f"  valid_rate:  {vr:.6f}   (structurally valid / n)\n"
            f"  n:           {n}"
        )

        trace = [ln.strip() for ln in raw_output.splitlines()
                 if ln.strip().startswith(("CD_SAMPLE", "CD_PROGRESS"))]
        if trace:
            feedback = (feedback + "\n" + "\n".join(trace[-4:])) if feedback else "\n".join(trace[-4:])

        if not feedback:
            feedback = raw_output[-3000:]

        return ParseResult(feedback=feedback, metrics=metrics)
