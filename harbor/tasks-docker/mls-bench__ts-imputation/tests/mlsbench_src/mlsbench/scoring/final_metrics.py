"""Single-final-line mitigation, NOT independent evaluation or a sandbox.

This blocks append/atexit overwrites. A worker that fabricates its only metric
line and exits successfully can still cheat.
"""
from __future__ import annotations

import re


def finalized_log(raw_output: str, *, max_metric_lines: int = 1) -> str:
    if type(max_metric_lines) is not int or max_metric_lines < 1:
        raise ValueError("max_metric_lines must be a positive integer")
    lines = raw_output.splitlines()
    finals = [line for line in lines if line.startswith("FINAL_TEST_METRICS:")]
    if len(finals) != 1:
        raise ValueError(f"expected one shell final metric line, found {len(finals)}")
    payload = finals[0][len("FINAL_TEST_METRICS:"):].strip()
    if not payload or payload.startswith("INVALID"):
        raise ValueError(payload or "empty shell final metric line")
    original = [line[len("TEST_METRICS:"):].strip() for line in lines
                if line.startswith("TEST_METRICS:")]
    if max_metric_lines == 1:
        if original != [payload]:
            raise ValueError("missing, duplicate or inconsistent original metric line")
    else:
        # Opt-in only for fixed evaluators that emit disjoint condition groups.
        # Never implement a last-match-wins merge, including for nan/inf tokens
        # which the original numeric parser may ignore.
        if not 1 <= len(original) <= max_metric_lines or any(not p for p in original):
            raise ValueError("missing or excessive original metric records")
        keys = []
        for record in original:
            names = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=", record)
            if not names:
                raise ValueError("metric record contains no assignments")
            keys.extend(names)
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate metric key; refusing overwrite")
        if " ".join(payload.split()) != " ".join(" ".join(original).split()):
            raise ValueError("inconsistent combined final metric record")
    # Preserve protocol checks (e.g. BUDGET_TRACE), never other metric lines.
    diagnostics = [line for line in lines if "TEST_METRICS:" not in line]
    return "\n".join(diagnostics + ["TEST_METRICS: " + payload]) + "\n"


def require_shell_final_metrics(parser_type, *, max_metric_lines: int = 1):
    from mlsbench.agent.parsers import ParseResult

    class FinalizedParser(parser_type):
        def parse(self, cmd_label, raw_output):
            try:
                sanitized = finalized_log(raw_output, max_metric_lines=max_metric_lines)
            except ValueError as exc:
                return ParseResult(feedback=f"Invalid finalized evaluation: {exc}", metrics={})
            return super().parse(cmd_label, sanitized)

    FinalizedParser.__name__ = parser_type.__name__
    FinalizedParser.__qualname__ = parser_type.__qualname__
    FinalizedParser.__module__ = parser_type.__module__
    return FinalizedParser
