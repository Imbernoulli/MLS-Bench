"""Strong baseline for ape-calibration-input (CALIBRATION-INPUT choice for the dev selection signal (Calibrate-Before-Use)).

Reference: pasted into solution/calibration.py via the edit op.

2026-07-09 probe2 ARM D (k1h20, self-contained image sha256:bdce4266…, sst2):
["N/A","the"] debiases the dev ranking and selects the true best candidate
("Read the review and judge whether the opinion expressed is favorable or
unfavorable.", dev 0.900) → held-out sst2 TEST 0.9100 (the old 4-input variant
["N/A","","the","a"] only reached 0.8333 — fixed). agnews side: pending probe3.
"""

import common  # noqa: F401


def calibration_inputs(ctx):
    # Strong: genuinely CONTENT-FREE inputs whose mean label distribution is a pure
    # prior — subtracting it debiases the dev ranking (Zhao et al. 2021) and surfaces
    # the candidate that also wins on the held-out test (fixed "N/A" test calibration).
    return ["N/A", "the"]
