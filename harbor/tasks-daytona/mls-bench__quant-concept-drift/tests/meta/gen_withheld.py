#!/usr/bin/env python3
"""Split the qlib cn_data of the scored universe at the first test date.

Every setting of quant-concept-drift tests on or after 2016-01-01 (csi300_shifted:
2016-01-01..2018-12-31, csi300: 2017-01-01..2020-08-01, csi300_recent:
2019-01-01..2020-08-01), so the Harbor image keeps the data up to 2015-12-31 for
the scored universe -- every instrument ever listed in instruments/csi300.txt or
csi100.txt, plus the SH000300 benchmark -- and this directory carries the rest:

  qlib_withheld.tar.gz   calendars/day.txt and instruments/{csi300,csi100}.txt in
                         full, and features/<inst>/<field>.day.bin.tail = the bytes
                         after the visible prefix of each scored-universe bin
  qlib_withheld_manifest.json
                         the cut, the sha256 of every file the eval restores, and
                         the two aggregate hashes the image build asserts

qlib's .day.bin is [float32 start index][float32 values...] indexed on the
calendar, so the visible prefix of a bin with start index s keeps
max(0, min(n, K - s)) values, K = number of calendar days before the cut.
scripts/_withheld_qlib.py appends each tail to its visible prefix and checks the
sha256 before the eval runs: the scored data are byte-identical to the measured
protocol's. Instruments outside the scored universe are left as they are.

    python3 gen_withheld.py <cn_data dir> <out dir>

Source: bohanlyu2022/mlsbench-harbor-qlib@sha256:58f0c45df7cba61d9a2b72c6b4b08320cc4d333af6b9fb45de3cb3ec45c25f14
(/root/.qlib/qlib_data/cn_data).
"""
import hashlib
import io
import json
import struct
import sys
import tarfile
from pathlib import Path

CUT = "2016-01-01"
INDEX_FILES = ("csi300", "csi100")
BENCHMARK = "sh000300"


def universe(root: Path) -> list[str]:
    u = {BENCHMARK}
    for n in INDEX_FILES:
        for line in (root / "instruments" / f"{n}.txt").read_text().splitlines():
            if line.strip():
                u.add(line.split("\t")[0].lower())
    return sorted(i for i in u if (root / "features" / i).is_dir())


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    cal = (root / "calendars" / "day.txt").read_text().splitlines()
    K = sum(1 for d in cal if d < CUT)
    files, agg_full, agg_vis = {}, hashlib.sha256(), hashlib.sha256()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=9) as tar:
        def add(name: str, data: bytes) -> None:
            ti = tarfile.TarInfo(name); ti.size = len(data); ti.mtime = 0
            tar.addfile(ti, io.BytesIO(data))
        for rel in ["calendars/day.txt"] + [f"instruments/{n}.txt" for n in INDEX_FILES]:
            data = (root / rel).read_bytes(); add(rel, data)
            files[rel] = hashlib.sha256(data).hexdigest()
        for inst in universe(root):
            for f in sorted((root / "features" / inst).iterdir()):
                raw = f.read_bytes(); rel = f"features/{inst}/{f.name}"
                s = int(struct.unpack("<f", raw[:4])[0]); n = (len(raw) - 4) // 4
                keep = max(0, min(n, K - s))
                vis, tail = raw[: 4 + 4 * keep], raw[4 + 4 * keep:]
                add(rel + ".tail", tail)
                files[rel] = hashlib.sha256(raw).hexdigest()
                agg_full.update(rel.encode() + hashlib.sha256(raw).digest())
                agg_vis.update(rel.encode() + hashlib.sha256(vis).digest())
    (out / "qlib_withheld.tar.gz").write_bytes(buf.getvalue())
    manifest = {"cut": CUT, "visible_calendar_days": K, "last_visible_day": cal[K - 1],
                "universe_size": len(universe(root)),
                "agg_sha256_full": agg_full.hexdigest(), "agg_sha256_visible": agg_vis.hexdigest(),
                "sha256": files}
    (out / "qlib_withheld_manifest.json").write_text(json.dumps(manifest, indent=0) + "\n")
    print({k: v for k, v in manifest.items() if k != "sha256"}, len(files), "files")


if __name__ == "__main__":
    main()
