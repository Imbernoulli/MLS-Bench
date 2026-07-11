"""Weak baseline for ape-calibration-input (CALIBRATION-INPUT choice for the dev selection signal (Calibrate-Before-Use)).

Reference: pasted into solution/calibration.py via the edit op.

2026-07-09 probe2 ARM D (k1h20, self-contained image sha256:bdce4266…, sst2):
this LABEL-BIASED gushing input mis-ranks the fixed pool and selects a distractor
("Answer the same thing for every review.", dev 0.555) → held-out sst2 TEST 0.7600
under the fixed "N/A" test calibration. agnews side: pending probe3 confirmation.
"""

import common  # noqa: F401


def calibration_inputs(ctx):
    # Weak: a LABEL-BIASED gushing sentence as the "content-free" input — it injects a
    # positive label prior into the dev calibration, so the ranking picks a distractor.
    return ["This review is wonderful. It is positive."]
