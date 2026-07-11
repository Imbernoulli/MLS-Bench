"""pending_full_market1501_anchors

The historical anchors used a 40-identity subset, ResNet-18, and 200 updates.
They are invalid for the complete Market-1501 / ResNet-50 / 60-epoch protocol.
Until this research surface has fresh full-protocol anchors, every parser-valid
result scores exactly zero. Verification failures also remain exact zero.
"""
from mlsbench.scoring.dsl import *

for _setting in ("easy", "medium", "hard"):
    term(
        f"map_{_setting}_pending",
        col(f"map_{_setting}").higher().id().sigmoid(
            floor=const(1.0), scale=1.0
        ),
    )
    term(
        f"rank1_{_setting}_pending",
        col(f"rank1_{_setting}").higher().id().sigmoid(
            floor=const(1.0), scale=1.0
        ),
    )
    setting(
        _setting,
        weighted_mean(
            (f"map_{_setting}_pending", 1.0),
            (f"rank1_{_setting}_pending", 1.0),
        ),
    )

task(gmean("easy", "medium", "hard"))

