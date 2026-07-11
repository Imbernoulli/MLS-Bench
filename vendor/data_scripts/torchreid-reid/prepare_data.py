"""Prepare data + weights for the torchreid person-re-identification (reid-*) tasks.

Produces, under {data_root}/torchreid/:
  market1501_full/train/       all 12,936 official training images (751 identities)
  market1501_full/query/       all 3,368 official query images
  market1501_full/gallery/     all 19,732 official gallery images
  weights/resnet50_imagenet.pth  torchvision ImageNet ResNet-50 state_dict

All images keep the Market-1501 filename convention <pid>_c<cam>s...jpg, so pid
and camid are parsed from the filename at load time. The complete official train,
query, and gallery splits are copied; no identity or image subset is selected.

Requires network on the HOST. Set HF_ENDPOINT=https://hf-mirror.com and an
http(s) proxy if the default hub is unreachable.
"""
import argparse
import os
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

_PID = re.compile(r"([-\d]+)_c(\d+)")


def _download_market_zip(cache_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    zpath = hf_hub_download(
        repo_id="aveocr/Market-1501-v15.09.15.zip",
        filename="Market-1501-v15.09.15.zip",
        repo_type="dataset",
        local_dir=str(cache_dir),
    )
    return Path(zpath)


def _extract(zpath: Path, out: Path) -> Path:
    if not (out / "Market-1501-v15.09.15").exists():
        with zipfile.ZipFile(zpath) as z:
            z.extractall(out)
    root = out / "Market-1501-v15.09.15"
    if not root.exists():
        # some zips extract flat
        for d in out.iterdir():
            if d.is_dir() and (d / "bounding_box_train").exists():
                return d
    return root


def _by_pid(folder: Path):
    d = defaultdict(list)
    for img in sorted(folder.glob("*.jpg")):
        m = _PID.search(img.name)
        if not m:
            continue
        pid = int(m.group(1))
        d[pid].append(img)
    return d


def build_full(market_root: Path, out_dir: Path):
    train_dir = market_root / "bounding_box_train"
    gallery_dir = market_root / "bounding_box_test"
    query_dir = market_root / "query"
    for source, name in (
        (train_dir, "train"),
        (query_dir, "query"),
        (gallery_dir, "gallery"),
    ):
        shutil.copytree(source, out_dir / name, dirs_exist_ok=True)

    counts = {
        name: len(list((out_dir / name).glob("*.jpg")))
        for name in ("train", "query", "gallery")
    }
    expected = {"train": 12_936, "query": 3_368, "gallery": 19_732}
    if counts != expected:
        raise RuntimeError(f"Market-1501 inventory mismatch: {counts} != {expected}")
    train_ids = {pid for pid in _by_pid(out_dir / "train") if pid >= 0}
    query_ids = {pid for pid in _by_pid(out_dir / "query") if pid >= 0}
    if len(train_ids) != 751 or len(query_ids) != 750:
        raise RuntimeError(
            f"Market-1501 identity mismatch: train={len(train_ids)} query={len(query_ids)}"
        )
    print(
        "REID_FULL train_ids=751 train_imgs=12936 query_ids=750 "
        "query_imgs=3368 gallery_imgs=19732",
        flush=True,
    )


def build_weights(weights_dir: Path):
    weights_dir.mkdir(parents=True, exist_ok=True)
    out = weights_dir / "resnet50_imagenet.pth"
    if out.exists():
        print(f"weights already present: {out}", flush=True)
        return
    import torch
    import torchvision

    # torchvision and torchreid ResNet-50 share conv/bn layer names, so the
    # checkpoint maps cleanly onto the
    # backbone (fc/classifier is skipped by shape mismatch at load time).
    m = torchvision.models.resnet50(weights="IMAGENET1K_V1")
    torch.save(m.state_dict(), out)
    print(f"WEIGHTS_BUILT {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "vendor/data"))
    args = ap.parse_args()

    root = Path(args.data_root) / "torchreid"
    root.mkdir(parents=True, exist_ok=True)
    full = root / "market1501_full"

    if (full / "train").exists() and (full / "query").exists() and (full / "gallery").exists():
        print("complete Market-1501 already present", flush=True)
    else:
        cache = root / "_cache"
        cache.mkdir(parents=True, exist_ok=True)
        zpath = _download_market_zip(cache)
        print(f"downloaded {zpath} ({zpath.stat().st_size/1e6:.1f} MB)", flush=True)
        market_root = _extract(zpath, cache)
        print(f"extracted -> {market_root}", flush=True)
        build_full(market_root, full)

    build_weights(root / "weights")
    print("REID_PREPARE_DONE", flush=True)


if __name__ == "__main__":
    main()
