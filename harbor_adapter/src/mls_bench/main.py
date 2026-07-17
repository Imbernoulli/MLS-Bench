"""CLI for the MLS-Bench → Harbor adapter.

Usage:
    uv run python -m mls_bench.main \
        [--output-dir ./datasets/mls-bench] \
        [--limit N] [--overwrite] [--task-ids t1 t2 ...] \
        [--mls-bench-root /path/to/MLS-Bench]
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from mls_bench.adapter import MlsBenchAdapter


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "datasets" / "mls-bench"


def _parse_task_ids(raw: list[str] | None) -> list[str] | None:
    if not raw:
        return None
    out: list[str] = []
    for item in raw:
        out.extend(t.strip() for t in item.split(",") if t.strip())
    return out or None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mls-bench-adapter")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help="Where to write generated Harbor task directories.")
    p.add_argument("--limit", type=int, default=None,
                   help="Generate at most N tasks (debug).")
    p.add_argument("--overwrite", action="store_true",
                   help="Wipe and re-render task dirs that already exist.")
    p.add_argument("--task-ids", nargs="+", default=None,
                   help="MLS-Bench task names to generate. Accepts either "
                        "space-separated names or comma-separated names. "
                        "Default: all 140.")
    p.add_argument("--mls-bench-root", type=Path, default=None,
                   help="Path to an MLS-Bench checkout. Auto-detected from cwd "
                        "if omitted.")
    p.add_argument("--continue-on-error", action="store_true",
                   help="Skip tasks that fail to render instead of aborting.")
    args = p.parse_args(argv)

    adapter = MlsBenchAdapter(
        output_dir=args.output_dir,
        limit=args.limit,
        overwrite=args.overwrite,
        task_ids=_parse_task_ids(args.task_ids),
        mls_bench_root=args.mls_bench_root,
        continue_on_error=args.continue_on_error,
    )
    try:
        result = adapter.run()
    except Exception:
        traceback.print_exc()
        return 1

    print(
        f"\nGenerated {result.generated}/{result.requested} tasks; "
        f"{len(result.failed)} failed.",
        file=sys.stderr,
    )
    if result.failed:
        print("Failures:", file=sys.stderr)
        for t, e in result.failed:
            print(f"  - {t}: {e}", file=sys.stderr)
    print(f"Dataset manifest: {args.output_dir / 'dataset.toml'}", file=sys.stderr)
    return 0 if not result.failed else 2


if __name__ == "__main__":
    sys.exit(main())
