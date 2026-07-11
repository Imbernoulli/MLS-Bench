"""Fail-closed parser for the three-condition prompt/postprocess protocol."""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


EXPECTED_N = 257
EXPECTED_TASK = "codegen-prompt-postprocess"
EXPECTED_SEED = 42
EXPECTED_PROTOCOL = "mbpp-sanitized-reserved-v2"
EXPECTED_MODEL_REVISION = "357b899b4714bf46d935fb9911e8139b5b9efc29"
EXPECTED_DATASET_REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
EXPECTED_PROBLEMS_SHA256 = "TO_BE_FILLED_BY_WORKER"
EXPECTED_PROGRESS = list(range(20, EXPECTED_N, 20))
CONDITIONS = ("direct", "fenced_wrapper", "trailing_text")

_PROTOCOL_RE = re.compile(
    r"^CG_PROTOCOL task=(\S+) protocol=(\S+) n=(\d+) seed=(\d+) "
    r"model_revision=([0-9a-f]{40}) dataset_revision=([0-9a-f]{40}) "
    r"problems_sha256=([0-9a-f]{64})$"
)
_ITEM_RE = re.compile(r"^CG_ITEM(?:\s+[a-z][a-z0-9_]*=\S+)+$")
_PROGRESS_RE = re.compile(r"^CG_PROGRESS(?:\s+[a-z][a-z0-9_]*=\S+)+$")
_METRIC_RE = re.compile(r"^CG_METRICS(?:\s+[a-z][a-z0-9_]*=\S+)+$")
_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_FAILURE_RE = re.compile(
    r"(?im)(?:\bFALLBACK\b|\bNONFINITE\b|"
    r"^\s*Traceback\s+\(most recent call last\):|"
    r"^\s*(?:ERROR|FATAL|CRITICAL)(?:\b|:)|"
    r"\b(?:verification|evaluation|generation|executor) (?:failed|failure)\b|"
    r"CUDA out of memory|OutOfMemoryError|No space left on device|"
    r"ModuleNotFoundError|FileNotFoundError|command not found|^\s*Killed\s*$)"
)


def _reject(raw_output: str, reason: str) -> ParseResult:
    return ParseResult(
        feedback=f"Three-condition MBPP verification rejected: {reason}\n{raw_output[-2000:]}",
        metrics={},
    )


def _pairs(line: str, prefix: str) -> dict[str, str] | None:
    pairs: dict[str, str] = {}
    for token in line.split()[1:]:
        if "=" not in token:
            return None
        key, value = token.split("=", 1)
        if not re.fullmatch(r"[a-z][a-z0-9_]*", key) or not value or key in pairs:
            return None
        pairs[key] = value
    return pairs if line.startswith(prefix + " ") else None


def _matches_rounded(value: float, total: int) -> bool:
    return abs(value - total / EXPECTED_N) <= 5.1e-7


class Parser(OutputParser):
    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        if cmd_label != "mbpp":
            return _reject(raw_output, f"unexpected command label {cmd_label!r}")
        if _FAILURE_RE.search(raw_output):
            return _reject(raw_output, "failure marker present")

        lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        protocol_rows = [
            (i, _PROTOCOL_RE.fullmatch(line))
            for i, line in enumerate(lines)
            if line.startswith("CG_PROTOCOL")
        ]
        item_rows = [
            (i, _ITEM_RE.fullmatch(line), line)
            for i, line in enumerate(lines)
            if line.startswith("CG_ITEM")
        ]
        progress_rows = [
            (i, _PROGRESS_RE.fullmatch(line), line)
            for i, line in enumerate(lines)
            if line.startswith("CG_PROGRESS")
        ]
        metric_rows = [
            (i, _METRIC_RE.fullmatch(line), line)
            for i, line in enumerate(lines)
            if line.startswith("CG_METRICS")
        ]

        if len(protocol_rows) != 1 or protocol_rows[0][1] is None:
            return _reject(raw_output, "expected one well-formed protocol proof")
        if len(item_rows) != EXPECTED_N or any(match is None for _, match, _ in item_rows):
            return _reject(raw_output, f"expected {EXPECTED_N} well-formed item proofs")
        if len(progress_rows) != len(EXPECTED_PROGRESS) or any(
            match is None for _, match, _ in progress_rows
        ):
            return _reject(raw_output, "progress proof is missing, duplicate, or malformed")
        if len(metric_rows) != 1 or metric_rows[0][1] is None:
            return _reject(raw_output, "expected one well-formed metric proof")
        if metric_rows[0][0] != len(lines) - 1:
            return _reject(raw_output, "metric proof is not the final output line")
        if not protocol_rows[0][0] < item_rows[0][0]:
            return _reject(raw_output, "protocol proof must precede item evaluation")

        protocol = protocol_rows[0][1]
        assert protocol is not None
        if not re.fullmatch(r"[0-9a-f]{64}", EXPECTED_PROBLEMS_SHA256):
            return _reject(raw_output, "canonical problem digest pin is not materialized")
        if (
            protocol.group(1) != EXPECTED_TASK
            or protocol.group(2) != EXPECTED_PROTOCOL
            or int(protocol.group(3)) != EXPECTED_N
            or int(protocol.group(4)) != EXPECTED_SEED
            or protocol.group(5) != EXPECTED_MODEL_REVISION
            or protocol.group(6) != EXPECTED_DATASET_REVISION
            or protocol.group(7) != EXPECTED_PROBLEMS_SHA256
        ):
            return _reject(raw_output, "task/protocol/model/dataset proof does not match the pin")

        expected_item_keys = {"i"} | {
            f"{condition}_{kind}"
            for condition in CONDITIONS
            for kind in ("pass", "parse")
        }
        totals = {
            condition: {"pass": 0, "parse": 0}
            for condition in CONDITIONS
        }
        item_bits = {
            condition: {"pass": [], "parse": []}
            for condition in CONDITIONS
        }
        item_positions: list[int] = []
        for expected_index, (line_index, _match, line) in enumerate(item_rows, start=1):
            pairs = _pairs(line, "CG_ITEM")
            if pairs is None or set(pairs) != expected_item_keys:
                return _reject(raw_output, f"item {expected_index} has unexpected fields")
            if not pairs["i"].isdigit() or int(pairs["i"]) != expected_index:
                return _reject(raw_output, "item proofs are missing, duplicated, or out of order")
            for condition in CONDITIONS:
                values = {}
                for kind in ("pass", "parse"):
                    raw_value = pairs[f"{condition}_{kind}"]
                    if raw_value not in {"0", "1"}:
                        return _reject(
                            raw_output,
                            f"item {expected_index} {condition}_{kind} is not boolean",
                        )
                    value = int(raw_value)
                    values[kind] = value
                    totals[condition][kind] += value
                    item_bits[condition][kind].append(value)
                if values["pass"] and not values["parse"]:
                    return _reject(
                        raw_output,
                        f"item {expected_index} passes {condition} but does not parse",
                    )
            item_positions.append(line_index)

        expected_progress_keys = {"completed", "total"} | {
            f"{condition}_passed" for condition in CONDITIONS
        }
        for offset, (line_index, _match, line) in enumerate(progress_rows):
            pairs = _pairs(line, "CG_PROGRESS")
            if pairs is None or set(pairs) != expected_progress_keys:
                return _reject(raw_output, "progress proof has unexpected fields")
            if not pairs["completed"].isdigit() or not pairs["total"].isdigit():
                return _reject(raw_output, "progress inventory fields are not integers")
            completed = int(pairs["completed"])
            total = int(pairs["total"])
            if completed != EXPECTED_PROGRESS[offset] or total != EXPECTED_N:
                return _reject(raw_output, "progress inventory/count is inconsistent")
            for condition in CONDITIONS:
                raw_value = pairs[f"{condition}_passed"]
                if not raw_value.isdigit() or int(raw_value) != sum(
                    item_bits[condition]["pass"][:completed]
                ):
                    return _reject(
                        raw_output,
                        f"{condition} progress count disagrees with item proofs",
                    )
            if not item_positions[completed - 1] < line_index < item_positions[completed]:
                return _reject(raw_output, "progress proof is out of order")

        metric_pairs = _pairs(metric_rows[0][2], "CG_METRICS")
        expected_metric_keys = {"task", "n", "elapsed"} | {
            f"{metric}_{condition}"
            for condition in CONDITIONS
            for metric in ("pass_at_1", "parse_rate")
        }
        if metric_pairs is None or set(metric_pairs) != expected_metric_keys:
            return _reject(raw_output, "metric proof has unexpected, missing, or duplicate fields")
        if not metric_pairs["n"].isdigit() or int(metric_pairs["n"]) != EXPECTED_N:
            return _reject(raw_output, f"metric proof does not report n={EXPECTED_N}")
        if metric_pairs["task"] != EXPECTED_TASK:
            return _reject(raw_output, "terminal metric proof has the wrong task identity")

        values = {}
        for key, raw_value in metric_pairs.items():
            if key in {"task", "n"}:
                continue
            if not _NUMBER_RE.fullmatch(raw_value):
                return _reject(raw_output, f"metric {key} is not a finite decimal")
            value = float(raw_value)
            if not math.isfinite(value):
                return _reject(raw_output, f"metric {key} is non-finite")
            values[key] = value
        if values["elapsed"] <= 0.0:
            return _reject(raw_output, "elapsed time must be positive")
        for condition in CONDITIONS:
            for metric, kind in (("pass_at_1", "pass"), ("parse_rate", "parse")):
                key = f"{metric}_{condition}"
                if not _matches_rounded(values[key], totals[condition][kind]):
                    return _reject(raw_output, f"{key} disagrees with item proofs")

        metrics = {
            f"{metric}_{condition}": values[f"{metric}_{condition}"]
            for condition in CONDITIONS
            for metric in ("pass_at_1", "parse_rate")
        }
        summary = ", ".join(
            f"{condition}={totals[condition]['pass']}/{EXPECTED_N}"
            for condition in CONDITIONS
        )
        return ParseResult(
            feedback=f"Full three-condition MBPP evaluation: {summary}",
            metrics=metrics,
        )
