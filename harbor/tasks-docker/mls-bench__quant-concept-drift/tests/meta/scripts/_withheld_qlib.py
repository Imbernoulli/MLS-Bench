"""Rebuild the full qlib cn_data for the eval; print the provider_uri to use.

Under Harbor the image keeps the scored universe's data only up to the first
test date (a `.mlsb_test_withheld` marker sits in cn_data) and the rest arrives
with tests/, mounted only at verification
($MLSBENCH_TASK_DIR/data/withheld/qlib_withheld.tar.gz). This writes a cn_data
under $OUTPUT_DIR whose calendar, csi300/csi100 instrument files and
scored-universe bins are restored in full (visible prefix + withheld tail, each
checked against qlib_withheld_manifest.json); every other file is a symlink to
the image's copy. Without the marker (native runs) it prints nothing, and the
workflow keeps the provider_uri of workflow_config.yaml.

    python _withheld_qlib.py <visible cn_data dir>
"""
import hashlib
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path


def main() -> int:
    vis = Path(sys.argv[1])
    if not (vis / ".mlsb_test_withheld").exists():
        return 0
    withheld = Path(os.environ.get("MLSBENCH_TASK_DIR", "")) / "data" / "withheld"
    tar_p, man_p = withheld / "qlib_withheld.tar.gz", withheld / "qlib_withheld_manifest.json"
    if not tar_p.is_file() or not man_p.is_file():
        sys.exit(f"_withheld_qlib: the withheld test-period data are not mounted ({tar_p})")
    want = json.loads(man_p.read_text())["sha256"]
    out = Path(os.environ.get("OUTPUT_DIR") or os.environ.get("TMPDIR") or "/tmp") / "mlsb_eval_data" / \
        os.environ.get("ENV", "run") / f"seed{os.environ.get('SEED', '42')}" / "cn_data"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    restored = {}
    with tarfile.open(tar_p, "r:gz") as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            data = tar.extractfile(m).read()
            if m.name.endswith(".tail"):
                rel = m.name[: -len(".tail")]
                data = (vis / rel).read_bytes() + data
            else:
                rel = m.name
            if hashlib.sha256(data).hexdigest() != want.get(rel):
                sys.exit(f"_withheld_qlib: restored {rel} does not match the original data")
            restored[rel] = data
    if set(restored) != set(want):
        sys.exit(f"_withheld_qlib: {len(set(want) - set(restored))} files missing from the withheld archive")
    # Mirror the visible tree: real dirs where something is restored, symlinks elsewhere.
    restored_dirs = {str(Path(r).parent) for r in restored}
    for dirpath, dirnames, filenames in os.walk(vis):
        rel_dir = os.path.relpath(dirpath, vis)
        rel_dir = "" if rel_dir == "." else rel_dir
        for d in list(dirnames):
            rd = os.path.join(rel_dir, d)
            if rd in restored_dirs or any(x.startswith(rd + "/") for x in restored_dirs):
                (out / rd).mkdir(exist_ok=True)
            else:
                (out / rd).symlink_to(vis / rd)
                dirnames.remove(d)
        for f in filenames:
            rf = os.path.join(rel_dir, f)
            if f == ".mlsb_test_withheld" or rf in restored:
                continue
            (out / rf).symlink_to(vis / rf)
    for rel, data in restored.items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_bytes(data)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
