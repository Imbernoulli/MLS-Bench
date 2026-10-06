#!/usr/bin/env python3
"""Split the scored TSLib CSVs into an agent-visible prefix and a withheld tail.

The Harbor image keeps only the prefix (config.json `harbor_extra_docker_steps`
truncates the files in place and drops a `.mlsb_test_withheld` marker); the
tail is staged from this directory into the bundle's tests/meta/, which Harbor
mounts only at verification. scripts/_withheld.sh concatenates prefix + tail
and checks the result against the original file's sha256 before the eval runs,
so the scored data are byte-identical to what the leaderboard was measured on.

Cut points follow TSLib's own split borders (data_provider/data_loader.py):
  * ETTh1 (Dataset_ETT_hour): train = 12 months, val = 4 months of hourly rows,
    so the visible prefix is the first 12*30*24 + 4*30*24 = 11520 data rows.
  * weather / electricity (Dataset_Custom): test = int(0.2 * n) final rows, so
    the visible prefix is the first n - int(0.2 * n) data rows.

    python3 gen_withheld.py <dir holding ETTh1.csv weather.csv electricity.csv> <out dir>

Source: bohanlyu2022/mlsbench-harbor-time-series-library@sha256:3dbf272d06b2fa2f8b789e4becc9b4f1590fefab9251ff0fbac477b0e8f78592
(/data/ETT-small/ETTh1.csv, /data/weather/weather.csv, /data/electricity/electricity.csv).
"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

SPEC = {"ETTh1.csv": "ett_hour", "weather.csv": "custom", "electricity.csv": "custom"}


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    sha_lines = []
    for name, kind in SPEC.items():
        raw = (src / name).read_bytes()
        lines = raw.splitlines(keepends=True)
        n = len(lines) - 1  # data rows (first line is the header)
        keep = 12 * 30 * 24 + 4 * 30 * 24 if kind == "ett_hour" else n - int(n * 0.2)
        prefix, tail = b"".join(lines[: keep + 1]), b"".join(lines[keep + 1:])
        assert prefix + tail == raw
        stem = name[: -len(".csv")]
        (out / f"{stem}_tail.csv.gz").write_bytes(gzip.compress(tail, compresslevel=9, mtime=0))
        manifest[name] = {
            "data_rows": n, "visible_rows": keep, "withheld_rows": n - keep,
            "sha256_full": hashlib.sha256(raw).hexdigest(),
            "sha256_visible": hashlib.sha256(prefix).hexdigest(),
            "bytes_full": len(raw), "bytes_visible": len(prefix),
        }
        sha_lines.append(f"{manifest[name]['sha256_full']}  {name}\n")
    (out / "withheld_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    (out / "withheld_sha256.txt").write_text("".join(sha_lines))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
