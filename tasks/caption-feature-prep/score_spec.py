"""Measured full-protocol Flickr8k calibration.

All ten research surfaces share the same official split, frozen CLIP/GPT-2
model, 7,500-step training protocol, and CIDEr/BLEU-4 evaluation.  The common
metric scale is calibrated by complete sample, greedy, and beam decoding runs.
Missing or invalid verifier metrics remain fail-closed at exact zero.
"""
from mlsbench.scoring.dsl import *

term(
    "cider_flickr",
    col("cider_flickr").higher().id().sigmoid(
        floor=const(0.245972),
        ref=const(0.586622),
    ),
)
term(
    "bleu4_flickr",
    col("bleu4_flickr").higher().id().sigmoid(
        floor=const(0.076101),
        ref=const(0.218874),
    ),
)
setting(
    "flickr",
    weighted_mean(
        ("cider_flickr", 1.0),
        ("bleu4_flickr", 1.0),
    ),
)
task(gmean("flickr"))
