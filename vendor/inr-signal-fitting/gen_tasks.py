#!/usr/bin/env python3
"""Retired INR task generator.

The former generator emitted broad ``fit_inr`` surfaces, directional task text,
placeholder anchors, concurrent setting groups, and permissive scripts. Running it
would overwrite the checked-in fail-closed task family. Task files are now reviewed
artifacts and must be changed explicitly with tests.
"""
from __future__ import annotations


MESSAGE = (
    "gen_tasks.py is intentionally disabled: it cannot regenerate the hardened INR "
    "tasks. Edit checked-in task artifacts explicitly and run "
    "tests/test_ship_inr_signal_fitting.py."
)


def main() -> int:
    print(MESSAGE)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
