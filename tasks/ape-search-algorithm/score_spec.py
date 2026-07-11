"""Pending exact-zero calibration for the full official APE protocol.

The former 300-row anchors are rejected. Until this surface has terminal full
AG News 7,600-row and SST-2 872-row measured anchors, every parser-valid result
maps to zero. Verifier failures also remain exact zero.
"""
from mlsbench.scoring.dsl import *

for _dataset in ("agnews", "sst2"):
    term(
        f"test_acc_{_dataset}_pending",
        col(f"test_acc_{_dataset}").higher().id().sigmoid(
            floor=const(1.0), scale=1.0
        ),
    )
    setting(
        _dataset,
        weighted_mean((f"test_acc_{_dataset}_pending", 1.0)),
    )

task(gmean("agnews", "sst2"))
