"""Strict completion parser for the official caption protocol."""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


EXPECTED_MODE = 'decoding'
EXPECTED_LABEL = "flickr"
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
RESULT = re.compile(
    rf"CAPTION_RESULT protocol=flickr8k_official_v1 mode=(\w+) "
    rf"train_images=(\d+) train_pairs=(\d+) eval_images=(\d+) "
    rf"epochs=(\d+) batch_size=(\d+) steps=(\d+) seed=(\d+) "
    rf"split_sha256=([0-9a-f]{{64}}) manifest_sha256=([0-9a-f]{{64}}) "
    rf"predictions_sha256=([0-9a-f]{{64}}) cider=({NUMBER}) "
    rf"bleu4=({NUMBER}) status=ok"
)
FAILURE_MARKERS = (
    "Traceback (most recent call last)",
    "CAPTION_FAILURE",
    "_FALLBACK",
    "VERIFICATION_FAILED",
    "CUDA out of memory",
    "OutOfMemoryError",
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
        lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        records = [line for line in lines if line.startswith("CAPTION_RESULT")]
        if cmd_label != EXPECTED_LABEL:
            return ParseResult(feedback="caption command label is invalid", metrics={})
        if len(records) != 1 or not lines or lines[-1] != records[0]:
            return ParseResult(feedback="caption verification did not complete", metrics={})
        if any(marker in raw_output for marker in FAILURE_MARKERS):
            return ParseResult(feedback="caption verification reported a failure", metrics={})
        match = RESULT.fullmatch(records[0])
        if match is None:
            return ParseResult(feedback="caption completion proof is malformed", metrics={})
        mode, train_images, train_pairs, eval_images, epochs, batch_size, steps, seed, _, _, _, raw_cider, raw_bleu = match.groups()
        observed = tuple(map(int, (train_images, train_pairs, eval_images, epochs, batch_size, steps, seed)))
        if mode != EXPECTED_MODE or observed != (6000, 30000, 1000, 10, 40, 7500, 42):
            return ParseResult(feedback="caption completion proof does not bind the required protocol", metrics={})
        cider, bleu = float(raw_cider), float(raw_bleu)
        if not math.isfinite(cider) or not math.isfinite(bleu) or not 0.0 <= cider <= 10.0 or not 0.0 <= bleu <= 1.0:
            return ParseResult(feedback="caption metrics are invalid", metrics={})
        return ParseResult(
            feedback=f"complete caption evaluation: CIDEr={cider:.6f}, BLEU-4={bleu:.6f}",
            metrics={"cider_flickr": cider, "bleu4_flickr": bleu},
        )
