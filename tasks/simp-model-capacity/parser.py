"""Strict completion parser for simp-model-capacity."""
from __future__ import annotations

import hashlib
import math
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


EXPECTED_TASK = 'simp-model-capacity'
EXPECTED_SURFACE = 'capacity'
PROTOCOL = 'gem-full-test-v2'
SETTINGS = ('asset', 'turk', 'wiki')
EXPECTED_COUNTS = {'asset': 359, 'turk': 359, 'wiki': 720}
SETTINGS_BINDING = 'asset:359,turk:359,wiki:720'
INVENTORY_SHA256 = '5089e28d4fbfd2b216106c249f1500e20f4bafa956e36ef93d9749674743a49e'
MODEL_DIGESTS = {'small_turk': '525b2890c4938d17e3b35012a685e7f906effdfa46a0c1e47351e3c10a48eec3', 'small_wikiauto': 'cc1da30dd2a1e39fb928892e63eb4fc57d1e28fc2dba106c1bdfc44c2795f3b8', 'base_turk': '696096d83e309dc7491ef59c74b530b45f45816aaf708c7394cc4c3181c4e261'}
NUMBER = r"[-+]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?"
METRIC = re.compile(
    rf"SIMP_METRICS protocol={re.escape(PROTOCOL)} "
    rf"task={re.escape(EXPECTED_TASK)} surface={re.escape(EXPECTED_SURFACE)} "
    rf"setting=(asset|turk|wiki) sari=({NUMBER}) bleu=({NUMBER}) "
    rf"n_sents=([0-9]+) plen=({NUMBER}) lenratio=({NUMBER})"
)
DONE = re.compile(
    rf"SIMP_DONE protocol={re.escape(PROTOCOL)} "
    rf"task={re.escape(EXPECTED_TASK)} surface={re.escape(EXPECTED_SURFACE)} "
    rf"settings={re.escape(SETTINGS_BINDING)} seed=([0-9]+) "
    rf"inventory_sha256=([0-9a-f]{64}) model=([a-z0-9_]+) "
    rf"model_sha256=([0-9a-f]{64}) metrics_sha256=([0-9a-f]{64}) "
    rf"elapsed=({NUMBER}) status=ok"
)
FAILURE_MARKERS = (
    "Traceback (most recent call last)",
    "SIMP_FAILURE",
    "SURFACE_ERROR",
    "SIMP_NONFINITE",
    "_FALLBACK",
    "CUDA out of memory",
    "OutOfMemoryError",
    "VERIFICATION_FAILED",
    "Command exited with code",
    "Segmentation fault",
    "Killed",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "CANCELLED",
    "NODE_FAIL",
    "[ERROR]",
)


class Parser(OutputParser):
    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        errors: list[str] = []
        lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        if cmd_label != "simplify":
            errors.append(f"unexpected command label {cmd_label!r}")
        for marker in FAILURE_MARKERS:
            if marker in raw_output:
                errors.append(f"failure marker {marker!r}")

        metric_lines = [line for line in lines if line.startswith("SIMP_METRICS")]
        done_lines = [line for line in lines if line.startswith("SIMP_DONE")]
        if len(metric_lines) != len(SETTINGS):
            errors.append(f"expected three metric records, got {len(metric_lines)}")
        if len(done_lines) != 1:
            errors.append(f"expected one completion record, got {len(done_lines)}")
        if not lines or not done_lines or lines[-1] != done_lines[0]:
            errors.append("completion record is not the unique final nonempty line")

        metrics: dict[str, float] = {}
        observed_settings: list[str] = []
        for index, line in enumerate(metric_lines):
            match = METRIC.fullmatch(line)
            if match is None:
                errors.append("malformed or misbound metric record")
                continue
            setting, sari_raw, bleu_raw, count_raw, plen_raw, ratio_raw = match.groups()
            observed_settings.append(setting)
            sari, bleu = float(sari_raw), float(bleu_raw)
            plen, ratio = float(plen_raw), float(ratio_raw)
            count = int(count_raw)
            if index >= len(SETTINGS) or setting != SETTINGS[index]:
                errors.append(f"metric settings are not in canonical order: {setting!r}")
            if count != EXPECTED_COUNTS[setting]:
                errors.append(f"wrong inventory count for {setting!r}: {count}")
            if not all(math.isfinite(value) for value in (sari, bleu, plen, ratio)):
                errors.append(f"non-finite metric for {setting!r}")
            elif (not 0.0 <= sari <= 100.0 or not 0.0 <= bleu <= 100.0
                  or not 0.0 <= plen <= 200.0 or not 0.0 <= ratio <= 10.0):
                errors.append(f"metric outside bounds for {setting!r}")
            else:
                metrics[f"sari_{setting}"] = sari
                metrics[f"bleu_{setting}"] = bleu
        if tuple(observed_settings) != SETTINGS:
            errors.append(f"incomplete or reordered setting inventory: {observed_settings}")

        if len(done_lines) == 1:
            match = DONE.fullmatch(done_lines[0])
            if match is None:
                errors.append("malformed or misbound completion record")
            else:
                seed_raw, inventory_sha, model_name, model_sha, metrics_sha, elapsed_raw = match.groups()
                if int(seed_raw) != 42:
                    errors.append(f"unexpected seed {seed_raw}")
                if inventory_sha != INVENTORY_SHA256:
                    errors.append("data inventory digest mismatch")
                if MODEL_DIGESTS.get(model_name) != model_sha:
                    errors.append("model identity digest mismatch")
                expected_metrics_sha = hashlib.sha256(
                    ("\n".join(metric_lines) + "\n").encode("utf-8")
                ).hexdigest()
                if metrics_sha != expected_metrics_sha:
                    errors.append("metric-record digest mismatch")
                elapsed = float(elapsed_raw)
                if not math.isfinite(elapsed) or elapsed <= 0.0:
                    errors.append("completion elapsed is not finite and positive")

        if errors:
            return ParseResult(
                feedback="Rejected simplification verification: " + "; ".join(errors),
                metrics={},
            )
        return ParseResult(
            feedback=(
                f"Complete {EXPECTED_TASK} verification on "
                f"{SETTINGS_BINDING} with strict terminal proof."
            ),
            metrics=metrics,
        )
