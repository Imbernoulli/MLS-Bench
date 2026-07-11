"""Strict atomic proof parser shared by full-resolution inpainting siblings."""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult

PROTOCOL = "places365-val256-fullres-v1"
DATASET_SHA256 = "24b4e639ef12a0012af525bc4cb443e4ab4aaea8369a1fb009b70e4a4aad5d48"
IMAGE_SIZE = 256
TRAIN_COUNT = 32_000
VAL_COUNT = 4_500
TRAIN_STEPS = 100_000
BATCH_SIZE = 8
PROGRESS_EVERY = 10_000
SETTINGS = ("small", "large", "strokes")
EXPECTED_TASK = "cv-inpaint-architecture"
EXPECTED_SURFACE = "arch"
MASK_RANGES = {
    "small": (0.06, 0.12),
    "large": (0.22, 0.38),
    "strokes": (0.14, 0.28),
}

HEX = r"[0-9a-f]{64}"
DEC = r"(?:0(?:\.\d+)?|[1-9]\d*(?:\.\d+)?)(?:[eE][+-]?\d+)?"
PROTOCOL_RE = re.compile(
    rf"INPAINT_PROTOCOL schema=(\d+) protocol=(\S+) dataset_sha256=({HEX}) "
    rf"image_size=(\d+) train_count=(\d+) val_count=(\d+) "
    rf"train_steps=(\d+) batch_size=(\d+) seed=(\d+) "
    rf"surface=(\S+) settings=(\S+) data_manifest_sha256=({HEX}) "
    rf"solution_sha256=({HEX}) parameters=(\d+)"
)
PROGRESS_RE = re.compile(rf"INPAINT_PROGRESS step=(\d+) train_loss=({DEC})")
SETTING_RE = re.compile(r"INPAINT_SETTING setting=(\S+) count=(\d+)")
ITEM_RE = re.compile(
    rf"INPAINT_ITEM setting=(\S+) index=(\d+) source_sha256=({HEX}) "
    rf"hole_pixels=(\d+) hole_abs_sum=({DEC}) hole_sq_sum=({DEC}) "
    rf"full_abs_sum=({DEC})"
)
METRICS_RE = re.compile(
    rf"INPAINT_METRICS surface=(\S+) setting=(\S+) count=(\d+) "
    rf"hole_l1=({DEC}) hole_psnr=({DEC}) full_l1=({DEC}) hole_frac=({DEC})"
)
COMPLETE_RE = re.compile(r"INPAINT_VERIFICATION scope=full status=ok")
FAILURE_RE = re.compile(
    r"(?:traceback|out of memory|\boom\b|killed|fallback|exception|error|failed|"
    r"\bnan\b|\binf\b)",
    re.IGNORECASE,
)


def _close(actual: float, expected: float) -> bool:
    tolerance = max(5.0e-8, 5.0e-7 * max(abs(actual), abs(expected)))
    return abs(actual - expected) <= tolerance


def _zero(reason: str) -> ParseResult:
    return ParseResult(
        feedback=f"{EXPECTED_TASK} verification rejected: {reason}",
        metrics={},
    )


class Parser(OutputParser):
    """Accept all three settings together or reject the entire run."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        try:
            if cmd_label != "full":
                return _zero(f"unexpected command label {cmd_label!r}")
            if FAILURE_RE.search(raw_output):
                return _zero("failure marker or non-finite token in output")

            lines = raw_output.splitlines()
            expected_line_count = (
                1
                + TRAIN_STEPS // PROGRESS_EVERY
                + len(SETTINGS) * (1 + VAL_COUNT + 1)
                + 1
            )
            if len(lines) != expected_line_count or any(not line for line in lines):
                return _zero(
                    f"proof line count is {len(lines)}; expected {expected_line_count}"
                )

            protocol_match = PROTOCOL_RE.fullmatch(lines[0])
            if protocol_match is None:
                return _zero("missing or malformed protocol proof")
            (
                schema,
                protocol,
                dataset_sha256,
                image_size,
                train_count,
                val_count,
                train_steps,
                batch_size,
                seed,
                surface,
                settings,
                _data_manifest_sha256,
                _solution_sha256,
                parameters,
            ) = protocol_match.groups()
            expected_protocol = (
                int(schema) == 1
                and protocol == PROTOCOL
                and dataset_sha256 == DATASET_SHA256
                and int(image_size) == IMAGE_SIZE
                and int(train_count) == TRAIN_COUNT
                and int(val_count) == VAL_COUNT
                and int(train_steps) == TRAIN_STEPS
                and int(batch_size) == BATCH_SIZE
                and int(seed) == 42
                and surface == EXPECTED_SURFACE
                and settings == ",".join(SETTINGS)
                and 0 < int(parameters) <= 120_000_000
            )
            if not expected_protocol:
                return _zero("protocol constants, surface, settings, or resources differ")

            cursor = 1
            for ordinal in range(1, TRAIN_STEPS // PROGRESS_EVERY + 1):
                match = PROGRESS_RE.fullmatch(lines[cursor])
                cursor += 1
                if match is None:
                    return _zero(f"malformed progress proof {ordinal}")
                step, loss = match.groups()
                value = float(loss)
                if int(step) != ordinal * PROGRESS_EVERY or not math.isfinite(value):
                    return _zero(f"inconsistent progress proof {ordinal}")

            output_metrics: dict[str, float] = {}
            reference_sources: list[str] | None = None
            feedback_parts: list[str] = []
            for expected_setting in SETTINGS:
                setting_match = SETTING_RE.fullmatch(lines[cursor])
                cursor += 1
                if (
                    setting_match is None
                    or setting_match.group(1) != expected_setting
                    or int(setting_match.group(2)) != VAL_COUNT
                ):
                    return _zero(f"missing or out-of-order {expected_setting} setting proof")

                sources: list[str] = []
                source_set: set[str] = set()
                hole_pixels: list[int] = []
                hole_abs: list[float] = []
                hole_sq: list[float] = []
                full_abs: list[float] = []
                min_fraction, max_fraction = MASK_RANGES[expected_setting]
                for expected_index in range(VAL_COUNT):
                    match = ITEM_RE.fullmatch(lines[cursor])
                    cursor += 1
                    if match is None:
                        return _zero(
                            f"malformed {expected_setting} item proof {expected_index}"
                        )
                    setting, index, source, pixels, abs_sum, sq_sum, total_abs = match.groups()
                    if setting != expected_setting or int(index) != expected_index:
                        return _zero(
                            f"out-of-order {expected_setting} item proof {expected_index}"
                        )
                    if source in source_set:
                        return _zero(
                            f"duplicate source in {expected_setting} item {expected_index}"
                        )
                    sources.append(source)
                    source_set.add(source)
                    pixel_count = int(pixels)
                    fraction = pixel_count / (IMAGE_SIZE * IMAGE_SIZE)
                    if not min_fraction <= fraction <= max_fraction:
                        return _zero(
                            f"mask range mismatch in {expected_setting} item {expected_index}"
                        )
                    values = tuple(float(value) for value in (abs_sum, sq_sum, total_abs))
                    if not all(math.isfinite(value) and value >= 0 for value in values):
                        return _zero(
                            f"invalid numeric proof in {expected_setting} item {expected_index}"
                        )
                    if values[0] > pixel_count * 3 + 1.0e-6:
                        return _zero(f"hole absolute sum out of range in item {expected_index}")
                    if values[1] > pixel_count * 3 + 1.0e-6:
                        return _zero(f"hole squared sum out of range in item {expected_index}")
                    if values[2] > IMAGE_SIZE * IMAGE_SIZE * 3 + 1.0e-6:
                        return _zero(f"full absolute sum out of range in item {expected_index}")
                    hole_pixels.append(pixel_count)
                    hole_abs.append(values[0])
                    hole_sq.append(values[1])
                    full_abs.append(values[2])

                if reference_sources is None:
                    reference_sources = sources
                elif sources != reference_sources:
                    return _zero("the three settings did not evaluate the same image inventory")

                metrics_match = METRICS_RE.fullmatch(lines[cursor])
                cursor += 1
                if metrics_match is None:
                    return _zero(f"missing {expected_setting} terminal metrics")
                metric_surface, metric_setting, count, *metric_values = metrics_match.groups()
                if (
                    metric_surface != EXPECTED_SURFACE
                    or metric_setting != expected_setting
                    or int(count) != VAL_COUNT
                ):
                    return _zero(f"{expected_setting} terminal metadata mismatch")
                hole_l1, hole_psnr, full_l1, hole_frac = map(float, metric_values)
                if not all(
                    math.isfinite(value)
                    for value in (hole_l1, hole_psnr, full_l1, hole_frac)
                ):
                    return _zero(f"{expected_setting} terminal metric is non-finite")

                total_pixels = sum(hole_pixels)
                denominator = total_pixels * 3
                recomputed_l1 = math.fsum(hole_abs) / denominator
                mse = math.fsum(hole_sq) / denominator
                recomputed_psnr = (
                    99.0 if mse < 1.0e-12 else 10.0 * math.log10(1.0 / mse)
                )
                recomputed_full = math.fsum(full_abs) / (
                    VAL_COUNT * 3 * IMAGE_SIZE * IMAGE_SIZE
                )
                recomputed_fraction = total_pixels / (VAL_COUNT * IMAGE_SIZE * IMAGE_SIZE)
                if not all(
                    _close(actual, expected)
                    for actual, expected in (
                        (hole_l1, recomputed_l1),
                        (hole_psnr, recomputed_psnr),
                        (full_l1, recomputed_full),
                        (hole_frac, recomputed_fraction),
                    )
                ):
                    return _zero(
                        f"{expected_setting} metrics do not recompute from item proof"
                    )
                if not (
                    0 <= hole_l1 <= 1
                    and 0 <= full_l1 <= 1
                    and 0 <= hole_psnr <= 99
                ):
                    return _zero(f"{expected_setting} metrics violate image bounds")
                output_metrics.update(
                    {
                        f"hole_l1_{expected_setting}": hole_l1,
                        f"hole_psnr_{expected_setting}": hole_psnr,
                        f"full_l1_{expected_setting}": full_l1,
                        f"hole_frac_{expected_setting}": hole_frac,
                    }
                )
                feedback_parts.append(
                    f"{expected_setting}: L1={hole_l1:.6f}, PSNR={hole_psnr:.3f}"
                )

            if cursor != len(lines) - 1 or COMPLETE_RE.fullmatch(lines[cursor]) is None:
                return _zero("missing terminal zero-exit completion proof")
            return ParseResult(
                feedback=(
                    f"{EXPECTED_TASK} full-resolution checkpoint; "
                    + "; ".join(feedback_parts)
                ),
                metrics=output_metrics,
            )
        except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
            return _zero(f"proof parsing failed: {exc}")
