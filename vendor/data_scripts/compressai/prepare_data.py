"""Prepare the Kodak image set for the CompressAI (compress-*) MLS-Bench tasks.

Downloads the 24 standard Kodak PNG images (kodim01..kodim24, 768x512) and writes:
  1. {data_root}/compressai/kodak/manifest.json — the ORIGINAL FIXED split (train
     kodim01..20, eval kodim21..24), used unchanged by the 3 pre-existing tasks
     (compress-entropy-model, compress-quantization-surrogate, compress-rd-target).
  2. {data_root}/compressai/kodak/{low,mid,high}/manifest.json — a FIXED 3-way
     split of the SAME 24 images by texture complexity (mean gradient magnitude of
     the luma channel; low/mid/high terciles, 8 images each: 6 train + 2 eval),
     used by the newer compress-* architecture-axis tasks so every task aggregates
     over >=3 genuinely different settings. Deterministic (content-based, no RNG).

Produces {data_root}/compressai/kodak/: kodim01.png..kodim24.png + manifest.json
  + low/*.png + low/manifest.json + mid/... + high/...
Requires network on the HOST (use the proxy on the dev machine).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from urllib.request import urlopen, Request

# r0k.us blocks proxied requests (403). This mirror contains byte-identical copies
# of the standard files; pin the Git commit and verify every PNG by SHA-256.
SOURCE_REPO = "MohamedBakrAli/Kodak-Lossless-True-Color-Image-Suite"
SOURCE_COMMIT = "dd2e1105fe8d9afe1120cd7e4c0e5e41f509a81b"
BASE = (
    "https://raw.githubusercontent.com/"
    f"{SOURCE_REPO}/{SOURCE_COMMIT}/PhotoCD_PCD0992"
)
N_IMAGES = 24
EVAL_IDS = [21, 22, 23, 24]  # fixed 4-image Kodak eval subset (original split)
SOURCE_MANIFEST = "source_manifest.json"

# SHA-256 of the exact 24 Git blobs at SOURCE_COMMIT. The local files also match
# that commit's Git blob IDs, independently confirming the pinned mirror inventory.
KODAK_SHA256 = {
    1: "a56e27cbf5f843c048b6af1d6e090760e9c92fadba88b7dee0205918a37523bd",
    2: "4f4b74a79237e311d72cad958237b5f7088d8bce1c82305ebefe1a70e3022dfd",
    3: "e25ca1ff2f0c0cb5fdfd5f9b0a0bb21ac4c3de3c84a67f35b09a85d3306249db",
    4: "e3b946107c5d3441c022f678d0c3caf1e224d81b1604ba840a4f88e562de61aa",
    5: "10349e963c5c813d327852f82c1795fa4148d69fedffc4c589bee458e3ac3d53",
    6: "363510303b715d4cbc384e1ce227e466b613a09e1b71ae985882bf8e7fbd9b18",
    7: "b77d3f006f42414bb242222e0482e750c0fb9e5ee8d4bed2f6f11c5605fe54a4",
    8: "ba23983c76b4832ee0e8af0592664756841a16779acd69f792e268fb6d13d6e7",
    9: "6a4361c2fc194feb4edaa9f9a4a0620fb9943e460ac7fdf037fb0f6dd6607a7d",
    10: "9dfb70f5867c29ff9ed6313683f19b3d867849e40fbc0c4c54a4a89df341cf23",
    11: "7936814b58b5387fce2e4e2488b4ec830dadd95fa9520f358ddb30990b50f2b6",
    12: "d78c37c2f04f23761ed2367dd77e2db584ddd4c3950833fecf89f199a8126980",
    13: "bc34a3ce58dea09dce1704c997171602de90cb34d0c8503a988b77f473d39b08",
    14: "55a94550ff18f3246c4074fd32b77b0c74447c26b6ad274d564d999c0450ba6e",
    15: "7538cbb80cb9103606c48b806eae57d56c885c7f90b9b3be70a41160f9cbb683",
    16: "a89c7268ccd4718ba424a99fc4643c572cf692ca6eae887185ceb4e9f11d2e54",
    17: "37afcc89fbdcb76d9518e04b2fc011027e2f4cd14b3b2f83cefd721641a47c5b",
    18: "1a9258c365988961d87a0598725b609139c303ad48a5aad6c503c3b1a87849aa",
    19: "b7450b264b1b0a411390d8931b112c27905a992520fc90569dc4b920aa32bbdc",
    20: "3b46c71e3b92a563820ba32936be8330c586c41f938efd94be938386aae4328a",
    21: "ac958597c82073f6bb65129c68f72b651db5b9efd82e11547d07350214bc268b",
    22: "1cee58eb1f2d9c7ebb254d208a03c783ce6cf2c4d8c2cf45e235dd23b4ce1b29",
    23: "e3111a2fd4da24af15d6459ef9eacfe54106b38e27b4a21821b75c3f5d2d5baf",
    24: "1071c68372cc5a01435c2c225a5cf7d4bb803846ec08bb6b3d6721b156d7cb96",
}

# FIXED 3-way content-complexity split (by mean-gradient-magnitude tercile of the
# 24 Kodak images; computed once, hard-coded here for determinism / no extra deps).
SPLITS = {
    "low":  {"train": [2, 3, 9, 10, 12, 15], "eval": [20, 23]},
    "mid":  {"train": [4, 7, 11, 16, 17, 19], "eval": [21, 22]},
    "high": {"train": [1, 5, 6, 8, 13, 14], "eval": [18, 24]},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_png(path: Path, expected_sha256: str) -> bool:
    try:
        if not path.is_file() or _sha256(path) != expected_sha256:
            return False
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.mode == "RGB" and sorted(image.size) == [512, 768]
    except (OSError, ValueError):
        return False


def _flat_manifest() -> dict:
    train = [{"path": f"kodim{i:02d}.png"} for i in range(1, N_IMAGES + 1)
             if i not in EVAL_IDS]
    evaluation = [{"path": f"kodim{i:02d}.png"} for i in EVAL_IDS]
    return {"train": train, "eval": evaluation}


def _split_manifest(split: dict) -> dict:
    return {
        "train": [{"path": f"kodim{i:02d}.png"} for i in split["train"]],
        "eval": [{"path": f"kodim{i:02d}.png"} for i in split["eval"]],
    }


def _source_manifest() -> dict:
    return {
        "schema_version": 1,
        "dataset": "Kodak PhotoCD PCD0992",
        "source_repo": SOURCE_REPO,
        "source_commit": SOURCE_COMMIT,
        "source_path": "PhotoCD_PCD0992",
        "files_sha256": {
            f"kodim{i:02d}.png": KODAK_SHA256[i]
            for i in range(1, N_IMAGES + 1)
        },
        "flat_split": _flat_manifest(),
        "content_splits": {
            name: _split_manifest(split) for name, split in SPLITS.items()
        },
    }


def _prepared_is_valid(out: Path) -> bool:
    try:
        source = json.loads((out / SOURCE_MANIFEST).read_text())
        if source != _source_manifest():
            return False
        if json.loads((out / "manifest.json").read_text()) != _flat_manifest():
            return False
        for i in range(1, N_IMAGES + 1):
            if not _valid_png(out / f"kodim{i:02d}.png", KODAK_SHA256[i]):
                return False
        for name, split in SPLITS.items():
            split_dir = out / name
            if json.loads((split_dir / "manifest.json").read_text()) != _split_manifest(split):
                return False
            for i in split["train"] + split["eval"]:
                if not _valid_png(split_dir / f"kodim{i:02d}.png", KODAK_SHA256[i]):
                    return False
        return True
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def _download(idx: int, out: Path) -> None:
    name = f"kodim{idx:02d}.png"
    dst = out / name
    expected = KODAK_SHA256[idx]
    if _valid_png(dst, expected):
        return
    if dst.exists():
        dst.unlink()
    url = f"{BASE}/{idx:02d}.png"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=120) as r, open(dst, "wb") as f:
        f.write(r.read())
    if not _valid_png(dst, expected):
        dst.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded Kodak file failed SHA-256/image validation: {name}")
    print(f"  {name} -> {dst.stat().st_size // 1024} KB", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    args = ap.parse_args()

    out = Path(args.data_root) / "compressai" / "kodak"
    out.mkdir(parents=True, exist_ok=True)
    if _prepared_is_valid(out):
        print(f"Kodak data already present and fully validated at {out}", flush=True)
        return
    for i in range(1, N_IMAGES + 1):
        _download(i, out)

    (out / "manifest.json").write_text(
        json.dumps(_flat_manifest(), indent=1))
    print(f"KODAK_BUILT train={N_IMAGES - len(EVAL_IDS)} eval={len(EVAL_IDS)}", flush=True)

    for name, sp in SPLITS.items():
        split_dir = out / name
        split_dir.mkdir(parents=True, exist_ok=True)
        ids = sp["train"] + sp["eval"]
        for i in ids:
            fn = f"kodim{i:02d}.png"
            shutil.copy(out / fn, split_dir / fn)
        (split_dir / "manifest.json").write_text(
            json.dumps(_split_manifest(sp), indent=1))
        print(f"KODAK_SPLIT_BUILT {name} train={len(sp['train'])} eval={len(sp['eval'])}",
              flush=True)

    (out / SOURCE_MANIFEST).write_text(
        json.dumps(_source_manifest(), sort_keys=True, separators=(",", ":")) + "\n"
    )
    if not _prepared_is_valid(out):
        raise RuntimeError("prepared Kodak data failed post-build integrity validation")


if __name__ == "__main__":
    main()
