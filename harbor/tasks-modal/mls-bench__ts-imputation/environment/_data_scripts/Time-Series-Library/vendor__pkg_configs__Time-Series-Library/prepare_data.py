"""Prepare script for Time-Series-Library data dependencies.

Downloads the thuml/Time-Series-Library HF dataset to the host data root so
the runtime data_bind ({data_root}:/data) actually serves files. Without
this, the bind would mount an empty host dir over the SIF's /data and every
test would fail trying to read /data/<dataset>/<file>.

Idempotent: if all 13 expected files are already present under data_root,
exits without re-downloading. Run via:

    mlsbench data Time-Series-Library
    # or implicitly via `mlsbench build Time-Series-Library`
"""

import argparse
import sys
from pathlib import Path

EXPECTED_FILES = [
    "ETT-small/ETTh1.csv",
    "weather/weather.csv",
    "electricity/electricity.csv",
    "traffic/traffic.csv",
    "m4/Monthly-train.csv",
    "m4/Quarterly-train.csv",
    "m4/Yearly-train.csv",
    "PSM/train.csv",
    "MSL/MSL_train.npy",
    "SMAP/SMAP_train.npy",
    "EthanolConcentration/EthanolConcentration_TRAIN.ts",
    "FaceDetection/FaceDetection_TRAIN.ts",
    "Handwriting/Handwriting_TRAIN.ts",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    args = parser.parse_args()

    data_root: Path = args.data_root.expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    missing = [f for f in EXPECTED_FILES if not (data_root / f).exists()]
    if not missing:
        print(f"[Time-Series-Library] All {len(EXPECTED_FILES)} dataset files "
              f"present under {data_root}; skipping download.")
        return

    print(f"[Time-Series-Library] {len(missing)}/{len(EXPECTED_FILES)} files "
          f"missing — downloading thuml/Time-Series-Library to {data_root}")

    from huggingface_hub import snapshot_download
    snapshot_download(
        "thuml/Time-Series-Library",
        repo_type="dataset",
        local_dir=str(data_root),
    )

    still_missing = [f for f in EXPECTED_FILES if not (data_root / f).exists()]
    if still_missing:
        print(f"[Time-Series-Library] ERROR: still missing after download: "
              f"{still_missing}", file=sys.stderr)
        sys.exit(1)

    print(f"[Time-Series-Library] All {len(EXPECTED_FILES)} files verified.")


if __name__ == "__main__":
    main()
