#!/usr/bin/env python3
"""Static validation entry point for every active INR task and baseline edit.

This is a compatibility wrapper around ``validate_oracle.py``. Its default invocation
is read-only and does not use a GPU. Pass the explicit ``validate_oracle.py --run`` form
for a serial end-to-end measurement of one selected task.
"""
from __future__ import annotations

from validate_oracle import main


if __name__ == "__main__":
    raise SystemExit(main())
