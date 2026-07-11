#!/usr/bin/env python3
"""Retired confounded INR sweep.

Use ``sweep_new_anchors.py``. That tool evaluates the fixed verifier surfaces and
reports measured rankings without encoding an assumed candidate order. The old sweep
mixed SIREN initialization and frequency and therefore cannot support an initialization
claim.
"""
from __future__ import annotations


def main() -> int:
    print(
        "sweep_anchors.py is retired; use sweep_new_anchors.py with an explicit "
        "--task, --signal, and --output."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
