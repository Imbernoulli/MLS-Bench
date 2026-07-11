"""CALIBRATION-INPUT choice for the dev selection signal (Calibrate-Before-Use) surface (agent-editable) — automatic prompt optimization.

Frozen Qwen2.5-0.5B-Instruct, inference-only, calibrated zero-shot execution;
disjoint train proposal/selection and full official evaluation. You design ONLY the `calibration_inputs(ctx) -> list[str]` content-free-input chooser; everything else is
fixed. the candidate pool and the held-out TEST calibration are FIXED (test always uses content-free "N/A"); only the DEV selection calibration is yours. A label-leaking calibration mis-ranks candidates (picks a distractor), while content-free inputs (N/A/""/the/a) debias the ranking and surface the candidate that also wins on test. Keep it deterministic. Both complete official dataset settings run serially on one GPU.
Reference baselines: vendor/prompt-optimization-lab/baselines/
"""
from __future__ import annotations

import common  # noqa: F401


# === EDITABLE REGION — implement calibration_inputs below ===
def calibration_inputs(ctx):
    # Weak: a LABEL-BIASED gushing sentence as the "content-free" input — it injects a
    # positive label prior into the dev calibration, so the ranking picks a distractor.
    return ["This review is wonderful. It is positive."]
# === END EDITABLE REGION ===
