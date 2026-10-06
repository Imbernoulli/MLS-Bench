"""Data-only prediction transport, never a transport for self-reported scores.

Call this in the host/private parser, not in the editable interpreter.  A second
frame invalidates the submission: neither first-match nor last-match wins.
"""
from __future__ import annotations

import json


MARKER = "MLSB_PREDICTIONS_V1:"
MAX_FRAME_BYTES = 2 * 1024 * 1024


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate prediction field: {key}")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError(f"non-finite JSON number: {value}")


def read_prediction_frame(raw_output: str) -> dict:
    frames = [line[len(MARKER):].strip() for line in raw_output.splitlines()
              if line.startswith(MARKER)]
    if len(frames) != 1:
        raise ValueError(f"expected exactly one prediction frame, found {len(frames)}")
    if len(frames[0].encode("utf-8")) > MAX_FRAME_BYTES:
        raise ValueError("prediction frame exceeds byte limit")
    try:
        value = json.loads(frames[0], object_pairs_hook=_unique_object,
                           parse_constant=_nonfinite)
    except (ValueError, RecursionError) as exc:
        raise ValueError(f"invalid prediction JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("prediction frame must be an object")
    return value
