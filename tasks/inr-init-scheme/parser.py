"""Strict completion-proof parser shared by the inr-* tasks."""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


_NUMBER = r"[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?"
_DATA_RE = re.compile(
    r"^DATA_INFO signal=(\S+) res=(\d+) n_coords=(\d+) dev=(cuda(?::\d+)?)$"
)
_FINAL_STEP_RE = re.compile(
    rf"^STEP_METRICS label=(\S+) step=2000/2000 "
    rf"loss=({_NUMBER}) psnr=({_NUMBER})$"
)
_METRIC_RE = re.compile(
    rf"^INR_METRICS signal=(\S+) psnr=({_NUMBER}) "
    rf"res=(\d+) elapsed=({_NUMBER})$"
)
_DONE_RE = re.compile(
    r"^INR_DONE signal=(\S+) n_coords=(\d+) steps=(\d+) seed=(-?\d+)$"
)
_FAILURE_MARKER = re.compile(
    r"^(?:INR_(?:FAILED|FAILURE)\b|verification\s+(?:failed|failure)\b|"
    r"Traceback \(most recent call last\):|"
    r"(?:[A-Za-z_][A-Za-z0-9_]*Error|ERROR|Exception)(?::|\b))",
    re.IGNORECASE,
)


class Parser(OutputParser):
    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        data_rows: list[tuple[int, re.Match[str]]] = []
        final_steps: list[tuple[int, re.Match[str]]] = []
        metric_rows: list[tuple[int, re.Match[str]]] = []
        done_rows: list[tuple[int, re.Match[str]]] = []
        malformed = False

        lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        for index, line in enumerate(lines):
            if _FAILURE_MARKER.match(line):
                malformed = True
            if line.startswith("DATA_INFO"):
                match = _DATA_RE.fullmatch(line)
                malformed = malformed or match is None
                if match is not None:
                    data_rows.append((index, match))
            elif line.startswith("STEP_METRICS") and "step=2000/2000" in line:
                match = _FINAL_STEP_RE.fullmatch(line)
                malformed = malformed or match is None
                if match is not None:
                    final_steps.append((index, match))
            elif line.startswith("INR_METRICS"):
                match = _METRIC_RE.fullmatch(line)
                malformed = malformed or match is None
                if match is not None:
                    metric_rows.append((index, match))
            elif line.startswith("INR_DONE"):
                match = _DONE_RE.fullmatch(line)
                malformed = malformed or match is None
                if match is not None:
                    done_rows.append((index, match))

        if (
            malformed
            or len(data_rows) != 1
            or len(final_steps) != 1
            or len(metric_rows) != 1
            or len(done_rows) != 1
            or not (
                data_rows[0][0]
                < final_steps[0][0]
                < metric_rows[0][0]
                < done_rows[0][0]
            )
            or not lines
            or done_rows[0][0] != len(lines) - 1
        ):
            return ParseResult(
                feedback="Invalid or incomplete INR evaluation protocol.", metrics={}
            )

        data_signal, data_res_raw, data_count_raw, _device = data_rows[0][1].groups()
        _step_label, loss_raw, train_psnr_raw = final_steps[0][1].groups()
        metric_signal, psnr_raw, metric_res_raw, elapsed_raw = metric_rows[0][1].groups()
        done_signal, done_count_raw, done_steps_raw, done_seed_raw = done_rows[0][1].groups()

        data_res = int(data_res_raw)
        metric_res = int(metric_res_raw)
        data_count = int(data_count_raw)
        done_count = int(done_count_raw)
        done_steps = int(done_steps_raw)
        done_seed = int(done_seed_raw)
        loss, train_psnr, psnr, elapsed = (
            float(value)
            for value in (loss_raw, train_psnr_raw, psnr_raw, elapsed_raw)
        )

        if (
            data_signal != cmd_label
            or metric_signal != cmd_label
            or done_signal != cmd_label
            or data_res != 256
            or metric_res != 256
            or data_count != 65536
            or done_count != 65536
            or done_steps != 2000
            or done_seed != 0
            or not all(
                math.isfinite(value) for value in (loss, train_psnr, psnr, elapsed)
            )
            or loss < 0.0
            or elapsed <= 0.0
        ):
            return ParseResult(
                feedback="Invalid or inconsistent INR evaluation proof.", metrics={}
            )

        feedback = (
            f"Results ({cmd_label}):\n"
            f"  reconstruction PSNR: {psnr:.4f} dB\n"
            f"  final train PSNR: {train_psnr:.4f} dB\n"
            f"  resolution: {metric_res}x{metric_res}\n"
            f"  coordinates: {done_count}\n"
            f"  optimizer steps: {done_steps}\n"
            f"  elapsed: {elapsed:.1f}s"
        )
        return ParseResult(
            feedback=feedback, metrics={f"psnr_{cmd_label}": psnr}
        )
