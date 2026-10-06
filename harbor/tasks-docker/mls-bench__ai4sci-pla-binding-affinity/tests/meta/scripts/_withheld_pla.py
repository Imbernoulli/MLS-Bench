"""Rebuild a labelled PDBbind test set for the eval; print the --data-dir to use.

Under Harbor the image keeps test{2013,2016,2019}_data.pt with the `label` key
removed (a `.mlsb_test_withheld` marker sits in the data dir) and the labels
arrive with tests/, mounted only at verification
($MLSBENCH_TASK_DIR/data/withheld/pla_test_labels.json). This writes the
labelled test set plus symlinks to the train/valid files into a per-setting dir
under $OUTPUT_DIR, after checking that the stripped file's features are
byte-for-byte the original's (pla_withheld_manifest.json). Without the marker
(native runs, where the image carries the full files) it prints the data dir
it was given.

    python _withheld_pla.py <test2013|test2016|test2019> <data dir>
"""
import hashlib
import json
import os
import sys
from pathlib import Path


def fingerprint(items) -> str:
    import torch
    h = hashlib.sha256()
    for it in items:
        for k in sorted(it):
            if k == "label":
                continue
            v = it[k]
            h.update(k.encode())
            if torch.is_tensor(v):
                h.update(str(v.dtype).encode()); h.update(str(tuple(v.shape)).encode())
                h.update(v.contiguous().numpy().tobytes())
            else:
                h.update(repr(v).encode())
    return h.hexdigest()


def main() -> int:
    test_set, data_dir = sys.argv[1], Path(sys.argv[2])
    if not (data_dir / ".mlsb_test_withheld").exists():
        print(data_dir)
        return 0
    import torch
    withheld = Path(os.environ.get("MLSBENCH_TASK_DIR", "")) / "data" / "withheld"
    labels_p, manifest_p = withheld / "pla_test_labels.json", withheld / "pla_withheld_manifest.json"
    if not labels_p.is_file() or not manifest_p.is_file():
        sys.exit(f"_withheld_pla: the withheld test labels are not mounted ({labels_p})")
    labels = json.loads(labels_p.read_text())[test_set]
    want = json.loads(manifest_p.read_text())[test_set]
    items = torch.load(data_dir / f"{test_set}_data.pt", weights_only=False)
    if len(items) != want["n"] or len(labels) != want["n"]:
        sys.exit(f"_withheld_pla: {test_set} has {len(items)} items, expected {want['n']}")
    if any("label" in it for it in items):
        sys.exit(f"_withheld_pla: {test_set} in {data_dir} unexpectedly carries labels")
    if fingerprint(items) != want["feature_fingerprint"]:
        sys.exit(f"_withheld_pla: {test_set} features differ from the original data")
    for it, v in zip(items, labels):
        it["label"] = torch.tensor([v], dtype=torch.float32)
    out = Path(os.environ.get("OUTPUT_DIR") or os.environ.get("TMPDIR") or "/tmp") / "mlsb_eval_data" / \
        os.environ.get("ENV", test_set) / f"seed{os.environ.get('SEED', '42')}"
    out.mkdir(parents=True, exist_ok=True)
    for name in ("train_data.pt", "valid_data.pt"):
        link = out / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(data_dir / name)
    tmp = out / f"{test_set}_data.pt.part"
    torch.save(items, tmp)
    os.replace(tmp, out / f"{test_set}_data.pt")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
