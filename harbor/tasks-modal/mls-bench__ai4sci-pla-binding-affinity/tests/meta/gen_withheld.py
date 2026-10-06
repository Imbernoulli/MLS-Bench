#!/usr/bin/env python3
"""Record the PDBbind test labels that the Harbor image withholds.

Run inside bohanlyu2022/mlsbench-harbor-ehign_pla@sha256:b6c61ad0f6c4913c161ddefdfd21e88a2a91e3ed7af0606d9543b757f45dbeea:

    python3 gen_withheld.py /data <out dir>

For each test set it writes the per-item float32 `label` values in file order
(pla_test_labels.json), plus the sha256 of the original .pt and a fingerprint of
every non-label tensor/int of every item (pla_withheld_manifest.json). The image
keeps the .pt files with the `label` key removed (config.json
harbor_extra_docker_steps) and drops the test*.csv files (pdbid,-logKd/Ki).
scripts/_withheld_pla.py re-inserts the labels for the eval after checking that
the stripped file's fingerprint equals the original's, so the scored data are
the measured protocol's data.
"""
import hashlib
import json
import sys
from pathlib import Path

import torch

SETS = ("test2013", "test2016", "test2019")


def fingerprint(items) -> str:
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


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    labels, manifest = {}, {}
    for s in SETS:
        p = src / f"{s}_data.pt"
        items = torch.load(p, weights_only=False)
        labels[s] = [float(it["label"].reshape(-1)[0]) for it in items]
        assert all(it["label"].dtype == torch.float32 and tuple(it["label"].shape) == (1,) for it in items)
        manifest[s] = {"n": len(items), "sha256_original_pt": hashlib.sha256(p.read_bytes()).hexdigest(),
                       "feature_fingerprint": fingerprint(items),
                       "sha256_original_csv": hashlib.sha256((src / f"{s}.csv").read_bytes()).hexdigest()}
    (out / "pla_test_labels.json").write_text(json.dumps(labels) + "\n")
    (out / "pla_withheld_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
