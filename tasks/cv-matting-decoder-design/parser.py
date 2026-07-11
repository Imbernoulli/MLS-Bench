"""Strict parser shared by the full Composition-1K matting task family."""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


PROTOCOL = "composition1k-full-v1"
EXPECTED_TASK_ID = "cv-matting-decoder-design"
EXPECTED_SURFACE = "decoder"
TRIMAP_WIDTHS = {"medium": 6, "wide": 9, "xwide": 12}
SETTINGS = tuple(TRIMAP_WIDTHS)
RESULT_KEYS = {
    "protocol",
    "task",
    "surface",
    "setting",
    "trimap_width",
    "sad",
    "mse",
    "grad",
    "unk_frac",
    "train",
    "test",
    "iters",
    "seed",
}
FAILURE_MARKERS = (
    "_FALLBACK",
    "Traceback (most recent call last):",
    "CUDA out of memory",
    "OutOfMemoryError",
    "Segmentation fault",
    "verification failed",
    "evaluation did not complete",
)
RESULT_LINE = re.compile(r"^MATTING_RESULT(?:\s+[A-Za-z_][A-Za-z0-9_]*=\S+)+$")
COMPLETE_LINE = re.compile(
    r"^MATTING_COMPLETE\s+protocol=(\S+)\s+task=(\S+)\s+surface=(\S+)\s+"
    r"settings=(\d+)\s+"
    r"manifest_sha256=([0-9a-f]{64})$"
)
NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _invalid(raw_output: str, reason: str) -> ParseResult:
    return ParseResult(
        feedback=f"Invalid full Composition-1K result: {reason}\n{raw_output[-2000:]}",
        metrics={},
    )


def _pairs(line: str) -> dict[str, str] | None:
    values: dict[str, str] = {}
    for token in line.split()[1:]:
        key, value = token.split("=", 1)
        if key in values:
            return None
        values[key] = value
    return values


class Parser(OutputParser):
    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        if cmd_label != "composition1k":
            return _invalid(raw_output, f"unexpected command label {cmd_label!r}")
        if any(marker.lower() in raw_output.lower() for marker in FAILURE_MARKERS):
            return _invalid(raw_output, "failure or fallback marker present")

        lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        complete_indexes = [i for i, line in enumerate(lines) if line.startswith("MATTING_COMPLETE")]
        if len(complete_indexes) != 1 or complete_indexes[0] != len(lines) - 1:
            return _invalid(raw_output, "expected one terminal completion record")
        complete = COMPLETE_LINE.fullmatch(lines[-1])
        if not complete:
            return _invalid(raw_output, "malformed completion record")
        if (
            complete.group(1) != PROTOCOL
            or complete.group(2) != EXPECTED_TASK_ID
            or complete.group(3) != EXPECTED_SURFACE
            or int(complete.group(4)) != 3
        ):
            return _invalid(raw_output, "completion protocol/task/surface/count mismatch")
        manifest_sha256 = complete.group(5)

        result_lines = [line for line in lines[:-1] if line.startswith("MATTING_RESULT")]
        if len(result_lines) != len(SETTINGS):
            return _invalid(raw_output, "expected exactly three result records")
        parsed: dict[str, dict[str, float]] = {}
        for expected_setting, line in zip(SETTINGS, result_lines):
            if not RESULT_LINE.fullmatch(line):
                return _invalid(raw_output, "malformed result record")
            pairs = _pairs(line)
            if pairs is None or set(pairs) != RESULT_KEYS:
                return _invalid(raw_output, f"expected fields {sorted(RESULT_KEYS)}")
            if (
                pairs["protocol"] != PROTOCOL
                or pairs["task"] != EXPECTED_TASK_ID
                or pairs["surface"] != EXPECTED_SURFACE
                or pairs["setting"] != expected_setting
            ):
                return _invalid(raw_output, "result protocol/task/surface/setting mismatch")
            integers = {"train": 43_100, "test": 1_000, "iters": 100_000, "seed": 42}
            for key, expected in integers.items():
                if not pairs[key].isdigit() or int(pairs[key]) != expected:
                    return _invalid(raw_output, f"expected {key}={expected}")
            if (
                not pairs["trimap_width"].isdigit()
                or int(pairs["trimap_width"]) != TRIMAP_WIDTHS[expected_setting]
            ):
                return _invalid(raw_output, "result trimap width mismatch")
            values = {}
            for key in ("sad", "mse", "grad", "unk_frac"):
                if not NUMBER.fullmatch(pairs[key]):
                    return _invalid(raw_output, f"{key} is not a decimal")
                value = float(pairs[key])
                if not math.isfinite(value):
                    return _invalid(raw_output, f"{key} is non-finite")
                values[key] = value
            if any(values[key] < 0.0 for key in ("sad", "mse", "grad")):
                return _invalid(raw_output, "negative matting error")
            if not 0.0 < values["unk_frac"] < 1.0:
                return _invalid(raw_output, "invalid unknown-band fraction")
            parsed[expected_setting] = values
        unknown_fractions = [parsed[setting]["unk_frac"] for setting in SETTINGS]
        if unknown_fractions != sorted(unknown_fractions):
            return _invalid(raw_output, "unknown-band fractions contradict trimap widths")

        data_line = (
            f"MATTING_DATA protocol={PROTOCOL} task={EXPECTED_TASK_ID} "
            f"surface={EXPECTED_SURFACE} train=43100 test=1000 crop=320 "
            f"train_trimap_width=6 "
            f"manifest_sha256={manifest_sha256}"
        )
        data_records = [line for line in lines if line.startswith("MATTING_DATA")]
        if data_records != [data_line]:
            return _invalid(raw_output, "missing unique data-manifest proof")
        complete_records = [
            line for line in lines if line.startswith("MATTING_TRAIN_COMPLETE")
        ]
        if complete_records != ["MATTING_TRAIN_COMPLETE steps=100000 batch=8"]:
            return _invalid(raw_output, "missing unique training completion proof")
        train_steps = []
        train_indexes = []
        train_records = [
            (index, line)
            for index, line in enumerate(lines)
            if line.startswith("MATTING_TRAIN ")
        ]
        for index, line in train_records:
            match = re.fullmatch(
                r"MATTING_TRAIN\s+step=(\d+)\s+loss=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
                line,
            )
            loss = float(match.group(2)) if match else float("nan")
            if not match or not math.isfinite(loss) or loss < 0.0:
                return _invalid(raw_output, "malformed, negative, or non-finite training proof")
            train_indexes.append(index)
            train_steps.append(int(match.group(1)))
        if train_steps != [0, 20_000, 40_000, 60_000, 80_000, 99_999]:
            return _invalid(raw_output, "incomplete or duplicate training progress proof")
        device_records = [
            (index, line) for index, line in enumerate(lines) if line.startswith("DEVICE ")
        ]
        if len(device_records) != 1 or not re.fullmatch(
            r"DEVICE cuda torch=\S+", device_records[0][1]
        ):
            return _invalid(raw_output, "CUDA execution proof missing")

        data_index = lines.index(data_records[0])
        train_complete_index = lines.index(complete_records[0])
        result_indexes = [
            index for index, line in enumerate(lines) if line.startswith("MATTING_RESULT")
        ]
        proof_order = [
            device_records[0][0],
            data_index,
            *train_indexes,
            train_complete_index,
            *result_indexes,
            complete_indexes[0],
        ]
        if proof_order != sorted(proof_order):
            return _invalid(raw_output, "protocol proof records are out of order")

        metrics = {}
        for setting in SETTINGS:
            for key in ("sad", "mse", "grad", "unk_frac"):
                metrics[f"{key}_{setting}"] = parsed[setting][key]
        return ParseResult(
            feedback=(
                f"Full Composition-1K {EXPECTED_TASK_ID}/{EXPECTED_SURFACE}: "
                + ", ".join(f"{setting} SAD={parsed[setting]['sad']:.4f}" for setting in SETTINGS)
            ),
            metrics=metrics,
        )
