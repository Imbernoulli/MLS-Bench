"""Output parser for the mono3d-* monocular 3D object detection tasks.

The harness emits one metric line per run:
    MONO3D_METRICS task=<T> setting=<L> ap25=<A25> ap50=<A50> miou=<I> \
        depth_err=<Dz> yaw_err=<Yd> dim_err=<De> steps=<n> elapsed=<S>

We surface one leaderboard column per (metric, setting label):
    ap25_<label>   AP3D at 3D-IoU>=0.25 over the TEST slice (PRIMARY, HIGHER better)
    ap50_<label>   AP3D at 3D-IoU>=0.50 (HIGHER)
    miou_<label>   mean 3D IoU over the slice (HIGHER)
    depth_err_<label>  median abs center-depth error, metres (feedback; LOWER)
    yaw_err_<label>    mean abs yaw error, degrees (feedback; LOWER)
    dim_err_<label>    mean abs dimension error, metres (feedback; LOWER)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult

_METRIC_RE = re.compile(
    r"MONO3D_METRICS\s+task=(?P<task>\S+)\s+setting=(?P<setting>\S+)\s+"
    r"ap25=(?P<ap25>[-\d.eE+]+)\s+ap50=(?P<ap50>[-\d.eE+]+)\s+miou=(?P<miou>[-\d.eE+]+)\s+"
    r"depth_err=(?P<depth_err>[-\d.eE+]+)\s+yaw_err=(?P<yaw_err>[-\d.eE+]+)\s+"
    r"dim_err=(?P<dim_err>[-\d.eE+]+)"
)


class Parser(OutputParser):
    """Parser for the monocular-3D detection AP3D / 3D-IoU tasks."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        metrics: dict = {}
        feedback = ""

        for line in raw_output.splitlines():
            m = _METRIC_RE.search(line)
            if not m:
                continue
            ap25 = float(m.group("ap25"))
            ap50 = float(m.group("ap50"))
            miou = float(m.group("miou"))
            dz = float(m.group("depth_err"))
            yerr = float(m.group("yaw_err"))
            derr = float(m.group("dim_err"))
            metrics[f"ap25_{cmd_label}"] = ap25
            metrics[f"ap50_{cmd_label}"] = ap50
            metrics[f"miou_{cmd_label}"] = miou
            metrics[f"depth_err_{cmd_label}"] = dz
            metrics[f"yaw_err_{cmd_label}"] = yerr
            metrics[f"dim_err_{cmd_label}"] = derr
            feedback = (
                f"Results ({cmd_label}):\n"
                f"  AP3D@0.25:   {ap25:.4f}   (PRIMARY, higher better; 0 = constant/mean box)\n"
                f"  AP3D@0.50:   {ap50:.4f}   (higher better)\n"
                f"  mean 3D IoU: {miou:.4f}   (higher better)\n"
                f"  depth err:   {dz:.4f} m  (median abs center-depth error; lower better)\n"
                f"  yaw err:     {yerr:.4f} deg (lower better)\n"
                f"  dim err:     {derr:.4f} m  (lower better)"
            )

        trace = [
            ln.strip()
            for ln in raw_output.splitlines()
            if ln.strip().startswith(("SURFACE_ERROR", "SETTING_SLICE", "SETTING_WARN",
                                      "MODEL_BUILT", "DATA_LOADED"))
        ]
        if trace:
            tail = "\n".join(trace[-6:])
            feedback = (feedback + "\n" + tail) if feedback else tail

        if not feedback:
            feedback = raw_output[-3000:]

        return ParseResult(feedback=feedback, metrics=metrics)
