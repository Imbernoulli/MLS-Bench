"""Fail-closed parser for the pinned full MBPP reserved-assertion protocol."""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


EXPECTED_N = 257
EXPECTED_TASK = "codegen-best-of-n-select"
EXPECTED_SEED = 42
EXPECTED_PROTOCOL = "mbpp-sanitized-reserved-v2"
EXPECTED_MODEL_REVISION = "357b899b4714bf46d935fb9911e8139b5b9efc29"
EXPECTED_DATASET_REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
EXPECTED_PROBLEMS_SHA256 = "TO_BE_FILLED_BY_WORKER"
EXPECTED_PROGRESS = list(range(20, EXPECTED_N, 20))

# item field -> (terminal metric, kind)
TASK_DIAGNOSTICS = {
    "codegen-best-of-n-select": {"oracle": ("oracle_pass_at_1", "bool")},
    "codegen-decode-length": {
        "parsed": ("parse_rate", "bool"),
        "tokens": ("avg_token_cap", "tokens"),
    },
    "codegen-docstring-design": {"parsed": ("parse_rate", "bool")},
    "codegen-fewshot-priming": {"parsed": ("parse_rate", "bool")},
    "codegen-output-extract": {"parsed": ("parse_rate", "bool")},
    "codegen-sample-budget": {"samples": ("avg_samples", "samples")},
    "codegen-sampling-strategy": {"visible": ("visible_solve_rate", "bool")},
    "codegen-self-consistency": {
        "oracle": ("oracle_pass_at_1", "bool"),
        "survivors": ("mean_survivors", "count8"),
        "clusters": ("mean_clusters", "count8"),
        "cluster": ("top_cluster", "count8"),
        "agreement": ("agreement_rate", "bool"),
        "changed": ("changed_selection_rate", "bool"),
    },
    "codegen-self-repair": {
        "visible": ("visible_solve_rate", "bool"),
        "helped": ("repair_help_rate", "bool"),
    },
}

_PROTOCOL_RE = re.compile(
    r"^CG_PROTOCOL task=(\S+) protocol=(\S+) n=(\d+) seed=(\d+) "
    r"model_revision=([0-9a-f]{40}) dataset_revision=([0-9a-f]{40}) "
    r"problems_sha256=([0-9a-f]{64})$"
)
_ITEM_RE = re.compile(r"^CG_ITEM(?:\s+[a-z][a-z0-9_]*=\S+)+$")
_PROGRESS_RE = re.compile(
    r"^CG_PROGRESS completed=(\d+) total=(\d+) passed=(\d+)$"
)
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
        feedback=f"Full MBPP verification rejected: {reason}\n{raw_output[-2000:]}",
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

        diagnostic_spec = TASK_DIAGNOSTICS[EXPECTED_TASK]

        lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        protocol_rows = [(i, _PROTOCOL_RE.fullmatch(line)) for i, line in enumerate(lines)
                         if line.startswith("CG_PROTOCOL")]
        item_rows = [(i, _ITEM_RE.fullmatch(line), line) for i, line in enumerate(lines)
                     if line.startswith("CG_ITEM")]
        progress_rows = [(i, _PROGRESS_RE.fullmatch(line)) for i, line in enumerate(lines)
                         if line.startswith("CG_PROGRESS")]
        metric_rows = [(i, _METRIC_RE.fullmatch(line), line) for i, line in enumerate(lines)
                       if line.startswith("CG_METRICS")]

        if len(protocol_rows) != 1 or protocol_rows[0][1] is None:
            return _reject(raw_output, "expected one well-formed protocol proof")
        if len(item_rows) != EXPECTED_N or any(match is None for _, match, _ in item_rows):
            return _reject(raw_output, f"expected {EXPECTED_N} well-formed item proofs")
        if (len(progress_rows) != len(EXPECTED_PROGRESS)
                or any(match is None for _, match in progress_rows)):
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

        expected_item_keys = {"i", "passed"} | set(diagnostic_spec)
        totals = {"passed": 0, **{key: 0 for key in diagnostic_spec}}
        item_positions: list[int] = []
        for expected_index, (line_index, _match, line) in enumerate(item_rows, start=1):
            pairs = _pairs(line, "CG_ITEM")
            if pairs is None or set(pairs) != expected_item_keys:
                return _reject(raw_output, f"item {expected_index} has unexpected fields")
            if not pairs["i"].isdigit() or int(pairs["i"]) != expected_index:
                return _reject(raw_output, "item proofs are missing, duplicated, or out of order")
            item_values: dict[str, int] = {}
            for key in {"passed"} | set(diagnostic_spec):
                if not pairs[key].isdigit():
                    return _reject(raw_output, f"item {expected_index} field {key} is not an integer")
                value = int(pairs[key])
                kind = "bool" if key == "passed" else diagnostic_spec[key][1]
                if kind == "bool" and value not in {0, 1}:
                    return _reject(raw_output, f"item {expected_index} field {key} is not boolean")
                if kind == "samples" and not 1 <= value <= 8:
                    return _reject(raw_output, f"item {expected_index} sample count is outside [1,8]")
                if kind == "count8" and not 0 <= value <= 8:
                    return _reject(raw_output, f"item {expected_index} count is outside [0,8]")
                if kind == "tokens" and not 64 <= value <= 640:
                    return _reject(raw_output, f"item {expected_index} token cap is outside [64,640]")
                item_values[key] = value
                totals[key] += value
            if item_values["passed"] and "oracle" in item_values and not item_values["oracle"]:
                return _reject(raw_output, f"item {expected_index} passes but has no oracle survivor")
            if item_values["passed"] and "parsed" in item_values and not item_values["parsed"]:
                return _reject(raw_output, f"item {expected_index} passes but is not parsed")
            if item_values.get("helped", 0) and not item_values.get("visible", 0):
                return _reject(raw_output, f"item {expected_index} is helped without passing provided tests")
            if "survivors" in item_values:
                survivors = item_values["survivors"]
                clusters = item_values["clusters"]
                cluster = item_values["cluster"]
                if clusters > survivors or cluster > survivors:
                    return _reject(raw_output, f"item {expected_index} cluster proof exceeds survivors")
                if survivors == 0 and any(
                    item_values[key] for key in ("clusters", "cluster", "agreement", "changed")
                ):
                    return _reject(raw_output, f"item {expected_index} has clusters without survivors")
                if survivors and (
                    clusters < 1 or cluster < 1
                    or cluster + clusters - 1 > survivors
                    or cluster * clusters < survivors
                ):
                    return _reject(raw_output, f"item {expected_index} has inconsistent cluster coverage")
                if item_values["agreement"] != int(cluster > 1):
                    return _reject(raw_output, f"item {expected_index} agreement disagrees with cluster size")
                if item_values["changed"] and (not item_values["agreement"] or survivors < 2):
                    return _reject(raw_output, f"item {expected_index} changed selection without agreement")
            item_positions.append(line_index)

        if "oracle" in totals and totals["passed"] > totals["oracle"]:
            return _reject(raw_output, "selected-pass count exceeds oracle count")
        if "parsed" in totals and totals["passed"] > totals["parsed"]:
            return _reject(raw_output, "passing-program count exceeds parsed-program count")
        if "helped" in totals and totals["helped"] > totals["visible"]:
            return _reject(raw_output, "repair-help count exceeds provided-test pass count")
        if "samples" in totals and totals["samples"] != 4 * EXPECTED_N:
            return _reject(raw_output, "fixed total candidate budget was not consumed exactly")
        if "tokens" in totals and totals["tokens"] != 256 * EXPECTED_N:
            return _reject(raw_output, "fixed token-cap budget was not consumed exactly")

        progress_completed: list[int] = []
        for progress_offset, (line_index, match) in enumerate(progress_rows):
            assert match is not None
            completed, total, passed = map(int, match.groups())
            progress_completed.append(completed)
            if total != EXPECTED_N or completed != EXPECTED_PROGRESS[progress_offset]:
                return _reject(raw_output, "progress inventory/count is inconsistent")
            if passed != sum(
                int((_pairs(row[2], "CG_ITEM") or {})["passed"])
                for row in item_rows[:completed]
            ):
                return _reject(raw_output, "progress pass count disagrees with item proofs")
            if not item_positions[completed - 1] < line_index < item_positions[completed]:
                return _reject(raw_output, "progress proof is out of order")

        metric_pairs = _pairs(metric_rows[0][2], "CG_METRICS")
        expected_metric_keys = {
            "task", "pass_at_1", "n", "elapsed",
            *[metric for metric, _kind in diagnostic_spec.values()],
        }
        if metric_pairs is None or set(metric_pairs) != expected_metric_keys:
            return _reject(raw_output, "metric proof has unexpected, missing, or duplicate fields")
        if not metric_pairs["n"].isdigit() or int(metric_pairs["n"]) != EXPECTED_N:
            return _reject(raw_output, f"metric proof does not report n={EXPECTED_N}")
        if metric_pairs["task"] != EXPECTED_TASK:
            return _reject(raw_output, "terminal metric proof has the wrong task identity")

        values: dict[str, float] = {}
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
        if not _matches_rounded(values["pass_at_1"], totals["passed"]):
            return _reject(raw_output, "pass_at_1 disagrees with item proofs")
        for item_key, (metric_key, _kind) in diagnostic_spec.items():
            if not _matches_rounded(values[metric_key], totals[item_key]):
                return _reject(raw_output, f"metric {metric_key} disagrees with item proofs")

        metrics = {f"pass_at_1_{cmd_label}": values["pass_at_1"]}
        for metric_key, _kind in diagnostic_spec.values():
            metrics[f"{metric_key}_{cmd_label}"] = values[metric_key]
        return ParseResult(
            feedback=(
                f"Full pinned MBPP evaluation: problems={EXPECTED_N}, "
                f"passed={totals['passed']}, pass_at_1={values['pass_at_1']:.6f}, "
                f"elapsed={values['elapsed']:.1f}s"
            ),
            metrics=metrics,
        )
