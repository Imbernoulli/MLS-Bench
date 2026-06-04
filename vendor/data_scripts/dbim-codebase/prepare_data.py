#!/usr/bin/env python3
"""Prepare all data for dbim-codebase tasks (sampler + scheduler).

Downloads and organises:
  1. Datasets: edges2handbags, DIODE-256, ImageNet val (+ val10k subset)
  2. Model checkpoints (adapted for current PyTorch)
  3. FID reference statistics
  4. Inception TorchScript model for FID evaluator

Output layout (maps 1:1 to /workspace/dbim-codebase/assets inside container):
  {data_root}/dbim_data/
  ├── datasets/
  │   ├── edges2handbags/{train,val}
  │   ├── DIODE-256/train/
  │   ├── ImageNet/val/<n0144...>/<*.JPEG>     # 50000 images, no train/
  │   ├── val_faster_imagefolder_10k_fn.txt
  │   └── val_faster_imagefolder_10k_label.txt
  ├── ckpts/
  │   ├── e2h_ema_0.9999_420000_adapted.pt
  │   ├── diode_ema_0.9999_440000_adapted.pt
  │   └── imagenet256_inpaint_ema_0.9999_400000.pt
  ├── stats/
  │   ├── edges2handbags_ref_64_data.npz
  │   ├── fid_imagenet_256_val.npz          # generated from ImageNet val on first run
  │   └── diode_ref_256_data.npz
  ├── inception-2015-12-05.pt
  ├── pt_inception-2015-12-05-6726825d.pth
  └── torch_cache/
      └── hub/checkpoints/
          ├── resnet50-0676ba61.pth
          └── vgg16-397923af.pth

ImageNet source
---------------
We fetch only the ILSVRC2012 *validation* split (~6.3 GB) — the dbim-codebase
runtime never reads the train split: the only ImageNet pipeline (inpainting)
runs `--split test` against `InpaintingVal10kSubset`, which samples 10k images
from val using the bundled `val_faster_imagefolder_10k_*.txt` lists, and the
FID reference (`fid_imagenet_256_val.npz`) is also derived from val.

Default source: ``mlx-vision/imagenet-1k`` on the Hugging Face Hub
(non-gated, hosts a ``val.zip`` already arranged as ``<synset>/<image>.JPEG``,
i.e. the exact layout produced by the historical `valprep.sh` script). We use
``HF_ENDPOINT=https://hf-mirror.com`` by default for compute environments
without direct access to ``huggingface.co``.

If you have a licensed copy of the original tarballs, set
``IMAGENET_VAL_TAR=/path/ILSVRC2012_img_val.tar`` (and optionally
``IMAGENET_TRAIN_TAR=...``) and the script will use those instead of the HF
mirror download. Setting ``IMAGENET_TRAIN_TAR`` will additionally extract the
train split into ``ImageNet/train/`` (not used by the cv-dbm-sampler task at
runtime, but available for downstream training experiments).

Usage:
    python vendor/data_scripts/dbim-codebase/prepare_data.py --data-root vendor/data
    python vendor/data_scripts/dbim-codebase/prepare_data.py --data-root vendor/data --skip-imagenet
    IMAGENET_VAL_TAR=/path/ILSVRC2012_img_val.tar \
      python vendor/data_scripts/dbim-codebase/prepare_data.py --data-root vendor/data
"""

import argparse
import csv
import re
import os
import shutil
import subprocess
import time
import zipfile
from http.cookiejar import CookieJar
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener


def run_cmd(cmd, cwd=None):
    """Run a shell command, raising on failure."""
    print(f"  $ {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True, cwd=cwd)


def download_google_drive(file_id: str, dest: Path):
    """Download a public Google Drive file via gdown.

    Google Drive's confirm-token flow for large files (>~100 MB) gates on
    a virus-scan warning page whose markup has shifted multiple times; the
    in-tree HTTP fallback used to scrape the token from the HTML body and
    cookies, but the modern page no longer carries that token. gdown
    handles every variant in upstream, so we delegate to it. gdown is
    declared as a host_data_prepare_requirement in
    vendor/pkg_configs/dbim-codebase/config.json.
    """
    import gdown

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(url, str(tmp), quiet=False)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    if not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"gdown produced no output for Google Drive file {file_id}; the "
            "file may be private, deleted, or hit an unexpected confirmation "
            "page. Check the file at https://drive.google.com/file/d/"
            f"{file_id}/view"
        )
    tmp.replace(dest)


# ---------------------------------------------------------------------------
# 1. edges2handbags
# ---------------------------------------------------------------------------
def prepare_edges2handbags(datasets_dir: Path):
    """Download and extract edges2handbags dataset."""
    out_dir = datasets_dir / "edges2handbags"
    if out_dir.exists() and (out_dir / "train").exists() and (out_dir / "val").exists():
        print("[edges2handbags] Already exists, skipping")
        return

    print("[edges2handbags] Downloading...", flush=True)
    tarball = datasets_dir / "edges2handbags.tar.gz"
    run_cmd(
        "wget -q http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/edges2handbags.tar.gz"
        f" -O {tarball}"
    )

    print("[edges2handbags] Extracting...", flush=True)
    run_cmd(f"tar -xzf {tarball} -C {datasets_dir}")
    tarball.unlink(missing_ok=True)

    assert (out_dir / "train").exists(), "edges2handbags extraction failed"
    print("[edges2handbags] Done")


# ---------------------------------------------------------------------------
# 2. DIODE
# ---------------------------------------------------------------------------
def prepare_diode(datasets_dir: Path):
    """Download DIODE raw data and preprocess to DIODE-256."""
    out_dir = datasets_dir / "DIODE-256" / "train"
    if out_dir.exists() and len(list(out_dir.iterdir())) > 100:
        print("[DIODE] DIODE-256 already exists, skipping")
        return

    raw_dir = datasets_dir / "DIODE"
    raw_dir.mkdir(parents=True, exist_ok=True)

    files_to_download = [
        ("http://diode-dataset.s3.amazonaws.com/train.tar.gz", "train.tar.gz"),
        ("http://diode-dataset.s3.amazonaws.com/train_normals.tar.gz", "train_normals.tar.gz"),
        ("https://diode-1254389886.cos.ap-hongkong.myqcloud.com/data_list.zip", "data_list.zip"),
    ]

    def _normals_already_extracted() -> bool:
        # train_normals.tar.gz unpacks *_normal.npy files alongside the depth
        # files inside raw_dir/train/outdoor/scene_*/scan_*/. Probe for one to
        # avoid re-downloading the ~80 GB tarball after a partial-prep crash.
        outdoor = raw_dir / "train" / "outdoor"
        if not outdoor.exists():
            return False
        for npy in outdoor.glob("scene_*/scan_*/*_normal.npy"):
            return True
        return False

    for url, fname in files_to_download:
        dest = raw_dir / fname
        if fname == "data_list.zip":
            extracted_marker_fn = lambda: (raw_dir / "data_list").exists()
        elif fname == "train.tar.gz":
            extracted_marker_fn = lambda: (raw_dir / "train").exists()
        elif fname == "train_normals.tar.gz":
            extracted_marker_fn = _normals_already_extracted
        else:
            extracted_marker_fn = None

        if extracted_marker_fn and extracted_marker_fn():
            print(f"[DIODE] {fname}: already extracted, skipping download")
            continue

        if not dest.exists():
            print(f"[DIODE] Downloading {fname}...", flush=True)
            run_cmd(f"wget -q {url} -O {dest}")

        print(f"[DIODE] Extracting {fname}...", flush=True)
        if fname.endswith(".tar.gz"):
            run_cmd(f"tar -xzf {dest} -C {raw_dir}")
        elif fname.endswith(".zip"):
            run_cmd(f"unzip -qo {dest} -d {raw_dir}")
        dest.unlink(missing_ok=True)

    print("[DIODE] Preprocessing to DIODE-256...", flush=True)
    _preprocess_diode(raw_dir, datasets_dir)
    print("[DIODE] Done")


def _preprocess_diode(raw_dir: Path, datasets_dir: Path):
    """Replicate preprocess_depth.py from DiffusionBridge repo."""
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
    from PIL import Image

    img_size = 256
    split = "train"
    data_csv = raw_dir / "data_list" / "train_outdoor.csv"
    target_dir = datasets_dir / f"DIODE-{img_size}" / split
    target_dir.mkdir(parents=True, exist_ok=True)

    all_files = []
    with open(data_csv, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            if row[-1] == "Unavailable":
                continue
            all_files.append(row[0].split("/")[-1])

    print(f"  Processing {len(all_files)} images...", flush=True)

    def plot_depth_map(dm, validity_mask, name):
        validity_mask = validity_mask > 0
        MIN_DEPTH = 0.5
        MAX_DEPTH = min(300, np.percentile(dm, 99))
        dm = np.clip(dm, MIN_DEPTH, MAX_DEPTH)
        dm = np.log(dm, where=validity_mask)
        dm = np.ma.masked_where(~validity_mask, dm)
        cmap = plt.cm.get_cmap("jet")
        cmap.set_bad(color="black")
        norm = plt.Normalize(vmin=0, vmax=np.log(MAX_DEPTH + 1.01))
        image = cmap(norm(dm))
        plt.imsave(name, np.clip(image, 0.0, 1.0))

    def plot_normal_map(normal_map, name):
        normal_viz = normal_map[:, :, :]
        normal_viz = normal_viz + np.equal(
            np.sum(normal_viz, 2, keepdims=True), 0.0
        ).astype(np.float32) * np.min(normal_viz)
        normal_viz = (normal_viz - np.min(normal_viz)) / 2.0
        plt.imsave(name, np.clip(normal_viz, 0.0, 1.0))

    for i, file in enumerate(all_files):
        out_path = target_dir / file
        if out_path.exists():
            continue

        scene_id, scan_id = file.split("_")[0], file.split("_")[1]
        base_path = raw_dir / split / "outdoor" / f"scene_{scene_id}" / f"scan_{scan_id}"

        pil_image = (
            Image.open(base_path / file)
            .convert("RGB")
            .resize((img_size, img_size), Image.BICUBIC)
        )

        depth = np.load(str(base_path / (file[:-4] + "_depth.npy"))).squeeze().astype(np.float32)
        depth_mask = np.load(str(base_path / (file[:-4] + "_depth_mask.npy"))).astype(np.float32)
        normal = np.load(str(base_path / (file[:-4] + "_normal.npy"))).astype(np.float32)

        image_depth = cv2.resize(depth, dsize=(img_size, img_size), interpolation=cv2.INTER_NEAREST)
        image_depth_mask = cv2.resize(depth_mask, dsize=(img_size, img_size), interpolation=cv2.INTER_NEAREST)
        normal = cv2.resize(normal, dsize=(img_size, img_size), interpolation=cv2.INTER_NEAREST)

        pil_image.save(str(out_path))
        plot_depth_map(image_depth, image_depth_mask, str(target_dir / (file[:-4] + "_depth.png")))
        plot_normal_map(normal, str(target_dir / (file[:-4] + "_normal.png")))

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(all_files)}", flush=True)

    print(f"  Preprocessed {len(all_files)} images to {target_dir}")


# ---------------------------------------------------------------------------
# 3. ImageNet
# ---------------------------------------------------------------------------
# We only need the ILSVRC2012 *validation* split for cv-dbm-sampler:
#   * The Imagenet test cmd runs `--split test` against InpaintingVal10kSubset,
#     which samples 10k images from val using val_faster_imagefolder_10k_*.txt
#   * The FID reference (fid_imagenet_256_val.npz) is generated on first run
#     from val (see evaluation/fid_util.py compute_fid_ref_stat)
#   * The pre_edit (vendor/pkg_configs/dbim-codebase/pre_edit.py) reroutes the
#     train/val dataset slots to reuse the testset, so the codepath that would
#     have read assets/datasets/ImageNet/train is never exercised.
#
# Default source: ``mlx-vision/imagenet-1k`` on Hugging Face Hub. This is a
# non-gated public mirror that hosts a single ``val.zip`` (~6.7 GB) already
# laid out as ``val/<synset>/<image>.JPEG`` -- exactly the structure the
# historical ILSVRC2012 valprep.sh produces and the layout
# torchvision.datasets.ImageFolder expects. Overrideable via the
# IMAGENET_VAL_TAR env var if a host has a licensed local copy.

HF_VAL_REPO = "mlx-vision/imagenet-1k"
HF_VAL_FILE = "val.zip"
HF_DOWNLOAD_RETRIES = 5
HF_DOWNLOAD_MAX_WAIT = 60


def _hf_download_val_zip(dest: Path) -> Path:
    """Download mlx-vision/imagenet-1k val.zip via Hugging Face hub with retries.

    Uses HF_ENDPOINT=https://hf-mirror.com by default (huggingface.co is often
    unreachable on compute nodes). Override with HF_ENDPOINT=https://huggingface.co
    if direct access is fine.
    """
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from huggingface_hub import hf_hub_download

    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(1, HF_DOWNLOAD_RETRIES + 1):
        try:
            print(
                f"[ImageNet] hf_hub_download {HF_VAL_REPO}/{HF_VAL_FILE} "
                f"via {os.environ['HF_ENDPOINT']} (attempt {attempt}/"
                f"{HF_DOWNLOAD_RETRIES})...",
                flush=True,
            )
            local_path = hf_hub_download(
                repo_id=HF_VAL_REPO,
                filename=HF_VAL_FILE,
                repo_type="dataset",
                local_dir=str(dest.parent),
            )
            return Path(local_path)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            wait = min(HF_DOWNLOAD_MAX_WAIT, 5 * attempt)
            print(
                f"[ImageNet] HF download attempt {attempt}/"
                f"{HF_DOWNLOAD_RETRIES} failed: {exc}; retry in {wait}s",
                flush=True,
            )
            time.sleep(wait)
    raise RuntimeError(
        f"Failed to download {HF_VAL_REPO}/{HF_VAL_FILE} after "
        f"{HF_DOWNLOAD_RETRIES} attempts. If your network can't reach "
        f"{os.environ['HF_ENDPOINT']}, set HF_ENDPOINT to a reachable Hugging "
        f"Face mirror or pre-place a licensed ILSVRC2012_img_val.tar at "
        f"{dest.parent / 'ILSVRC2012_img_val.tar'} and re-run."
    ) from last_err


def _extract_val_zip(zip_path: Path, val_dir: Path) -> None:
    """Extract val.zip so synset dirs land directly under val_dir.

    The HF zip uses ``val/<synset>/<image>.JPEG`` paths internally. We strip
    the leading ``val/`` so the result is ``val_dir/<synset>/<image>.JPEG``.
    """
    val_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ImageNet] Extracting {zip_path.name} -> {val_dir}", flush=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        for i, info in enumerate(members):
            name = info.filename
            # All entries start with "val/"; strip exactly one prefix copy.
            if name.startswith("val/"):
                name = name[len("val/"):]
            if not name or name.endswith("/"):
                continue
            # Skip macOS AppleDouble junk this HF mirror's zip carries beside
            # every real image (a __MACOSX/ tree + ._<img>.JPEG stubs). They are
            # 212-byte non-images; ImageFolder would ingest them (they end in
            # .JPEG) and the first-run full-val FID-reference pass crashes on the
            # first one (PIL.UnidentifiedImageError) -> best_fid_Imagenet missing.
            _base = name.rsplit("/", 1)[-1]
            if "__MACOSX" in name.split("/") or _base.startswith("._") or _base == ".DS_Store":
                continue
            target = val_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)
            if (i + 1) % 5000 == 0:
                print(
                    f"  Extracted {i + 1}/{len(members)} entries...",
                    flush=True,
                )
    print(f"[ImageNet] Extracted {len(members)} entries", flush=True)


def _val_dir_ready(val_dir: Path, min_synsets: int = 1000, min_images: int = 50_000) -> bool:
    if not val_dir.exists():
        return False
    synsets = [p for p in val_dir.iterdir() if p.is_dir()]
    if len(synsets) < min_synsets:
        return False
    # Quick image count without walking everything: sample 5 synsets and
    # require all of them to be non-empty + total above min_images is hard
    # without a full walk, so we fall back to checking length only when below.
    total = sum(1 for _ in val_dir.glob("*/*.JPEG"))
    if total < min_images:
        return False
    return True


def _prune_macos_junk(val_dir: Path) -> None:
    """Remove macOS AppleDouble junk left by an unfiltered val.zip extraction
    (a ``__MACOSX/`` tree + ``._*.JPEG`` stubs beside every real image). Repairs
    a val/ dir that was staged before the extraction filter was added."""
    if not val_dir.exists():
        return
    macosx = val_dir / "__MACOSX"
    had_macosx = macosx.exists()
    if had_macosx:
        shutil.rmtree(macosx, ignore_errors=True)
    removed = 0
    for junk in val_dir.rglob("._*"):
        try:
            junk.unlink()
            removed += 1
        except OSError:
            pass
    if had_macosx or removed:
        print(f"[ImageNet] pruned macOS junk (__MACOSX={had_macosx}, ._* files={removed})", flush=True)


def prepare_imagenet(datasets_dir: Path):
    """Download ILSVRC2012 val split (only what cv-dbm-sampler needs).

    Source priority:
      1. IMAGENET_VAL_TAR env var (a host-provided ILSVRC2012_img_val.tar
         from a licensed copy) -- extract + valprep.sh.
      2. mlx-vision/imagenet-1k val.zip via HF mirror (default).

    The optional IMAGENET_TRAIN_TAR env var is still honoured if a user wants
    the full train set staged for downstream experiments, but it is *not*
    required for the cv-dbm-sampler smoke (the pre_edit reroutes the train
    slot to reuse val).
    """
    out_dir = datasets_dir / "ImageNet"
    train_dir = out_dir / "train"
    val_dir = out_dir / "val"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Repair: an earlier unfiltered extraction may have left macOS AppleDouble
    # junk (__MACOSX/ + ._*.JPEG) under val/. ImageFolder ingests these
    # 212-byte non-images and the full-val FID-reference pass crashes on the
    # first one. Prune before the readiness check so a dirty val gets repaired.
    _prune_macos_junk(val_dir)

    # --- val split ---------------------------------------------------------
    if _val_dir_ready(val_dir):
        print("[ImageNet] val/ already populated, skipping")
    else:
        val_tar_env = os.environ.get("IMAGENET_VAL_TAR")
        val_tar = Path(val_tar_env) if val_tar_env else None

        if val_tar and val_tar.exists():
            print(f"[ImageNet] Extracting licensed val tar {val_tar}...", flush=True)
            val_dir.mkdir(parents=True, exist_ok=True)
            run_cmd(f"tar -xf {val_tar} -C {val_dir}")
            run_cmd(
                "wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash",
                cwd=val_dir,
            )
        else:
            zip_path = datasets_dir / HF_VAL_FILE
            if not zip_path.exists() or zip_path.stat().st_size < 1024:
                zip_path = _hf_download_val_zip(zip_path)
            _extract_val_zip(zip_path, val_dir)
            # Free disk: the zip is ~6.7 GB and we don't need it once extracted.
            try:
                zip_path.unlink()
            except FileNotFoundError:
                pass

        if not _val_dir_ready(val_dir):
            raise RuntimeError(
                f"ImageNet val/ at {val_dir} did not pass the readiness check "
                "(>=1000 synset dirs, >=50,000 .JPEG files). Inspect the "
                "extraction output above."
            )
        print("[ImageNet] val/ ready")

    # --- optional train split (NOT required for cv-dbm-sampler) -----------
    train_tar_env = os.environ.get("IMAGENET_TRAIN_TAR")
    if train_tar_env:
        train_tar = Path(train_tar_env)
        if train_tar.exists() and (
            not train_dir.exists() or len(list(train_dir.iterdir())) < 1000
        ):
            print(
                f"[ImageNet] Extracting optional licensed train tar {train_tar}...",
                flush=True,
            )
            train_dir.mkdir(parents=True, exist_ok=True)
            run_cmd(f"tar -xf {train_tar} -C {train_dir}")
            run_cmd(
                'find . -name "*.tar" | while read NAME; do'
                ' mkdir -p "${NAME%.tar}";'
                ' tar -xf "${NAME}" -C "${NAME%.tar}";'
                ' rm -f "${NAME}";'
                " done",
                cwd=train_dir,
            )

    _download_val10k_files(datasets_dir)
    print("[ImageNet] Done")


def _download_val10k_files(datasets_dir: Path):
    """Download val_faster_imagefolder_10k_{fn,label}.txt from DiffusionBridge repo."""
    base_url = (
        "https://raw.githubusercontent.com/thu-ml/DiffusionBridge"
        "/92522733cc602686df77f07a1824bb89f89cda1a/assets/datasets"
    )
    for fname in [
        "val_faster_imagefolder_10k_fn.txt",
        "val_faster_imagefolder_10k_label.txt",
    ]:
        dest = datasets_dir / fname
        if dest.exists():
            print(f"[ImageNet] {fname} already exists, skipping")
            continue
        print(f"[ImageNet] Downloading {fname}...", flush=True)
        run_cmd(f"wget -q {base_url}/{fname} -O {dest}")


# ---------------------------------------------------------------------------
# 4. Checkpoints
# ---------------------------------------------------------------------------
def prepare_checkpoints(ckpts_dir: Path):
    """Download and adapt model checkpoints."""
    ckpts_dir.mkdir(parents=True, exist_ok=True)

    HF_BASE = "https://huggingface.co/alexzhou907/DDBM/resolve/main"

    # --- edges2handbags ---
    raw_e2h = ckpts_dir / "e2h_ema_0.9999_420000.pt"
    adapted_e2h = ckpts_dir / "e2h_ema_0.9999_420000_adapted.pt"
    if not adapted_e2h.exists():
        if not raw_e2h.exists():
            print("[ckpts] Downloading e2h checkpoint...", flush=True)
            run_cmd(f"wget -q {HF_BASE}/e2h_ema_0.9999_420000.pt -O {raw_e2h}")
        print("[ckpts] Adapting e2h checkpoint...", flush=True)
        _adapt_e2h(raw_e2h, adapted_e2h)
    else:
        print("[ckpts] e2h adapted checkpoint already exists, skipping")

    # --- DIODE ---
    raw_diode = ckpts_dir / "diode_ema_0.9999_440000.pt"
    adapted_diode = ckpts_dir / "diode_ema_0.9999_440000_adapted.pt"
    if not adapted_diode.exists():
        if not raw_diode.exists():
            print("[ckpts] Downloading diode checkpoint...", flush=True)
            run_cmd(f"wget -q {HF_BASE}/diode_ema_0.9999_440000.pt -O {raw_diode}")
        print("[ckpts] Adapting diode checkpoint...", flush=True)
        _adapt_diode(raw_diode, adapted_diode)
    else:
        print("[ckpts] diode adapted checkpoint already exists, skipping")

    # --- ImageNet inpainting ---
    imagenet_ckpt = ckpts_dir / "imagenet256_inpaint_ema_0.9999_400000.pt"
    if not imagenet_ckpt.exists():
        print("[ckpts] Downloading ImageNet inpainting checkpoint (Google Drive)...", flush=True)
        gdrive_id = "1WozJyVOAFukj0nUYLS-ZUp1-QHuGNfox"
        download_google_drive(gdrive_id, imagenet_ckpt)
    else:
        print("[ckpts] ImageNet checkpoint already exists, skipping")


def _adapt_e2h(raw_path: Path, adapted_path: Path):
    """Squeeze attention weights for e2h checkpoint (removes flash_attn dim)."""
    import torch
    sd = torch.load(raw_path, map_location="cpu")
    modules = []
    for i in range(5, 16):
        if i in (8, 12):
            continue
        modules.append(f"input_blocks.{i}.1.qkv.weight")
        modules.append(f"input_blocks.{i}.1.proj_out.weight")
    modules.append("middle_block.1.qkv.weight")
    modules.append("middle_block.1.proj_out.weight")
    for i in range(12):
        modules.append(f"output_blocks.{i}.1.qkv.weight")
        modules.append(f"output_blocks.{i}.1.proj_out.weight")
    for name in modules:
        sd[name] = sd[name].squeeze(-1)
    torch.save(sd, adapted_path)
    print(f"  Saved adapted checkpoint to {adapted_path}")


def _adapt_diode(raw_path: Path, adapted_path: Path):
    """Squeeze attention weights for diode checkpoint."""
    import torch
    sd = torch.load(raw_path, map_location="cpu")
    modules = []
    for i in range(10, 18):
        if i in (12, 15):
            continue
        modules.append(f"input_blocks.{i}.1.qkv.weight")
        modules.append(f"input_blocks.{i}.1.proj_out.weight")
    modules.append("middle_block.1.qkv.weight")
    modules.append("middle_block.1.proj_out.weight")
    for i in range(9):
        modules.append(f"output_blocks.{i}.1.qkv.weight")
        modules.append(f"output_blocks.{i}.1.proj_out.weight")
    for name in modules:
        sd[name] = sd[name].squeeze(-1)
    torch.save(sd, adapted_path)
    print(f"  Saved adapted checkpoint to {adapted_path}")


# ---------------------------------------------------------------------------
# 5. FID reference statistics
# ---------------------------------------------------------------------------
def prepare_stats(stats_dir: Path):
    """Download FID reference statistics."""
    stats_dir.mkdir(parents=True, exist_ok=True)

    HF_BASE = "https://huggingface.co/alexzhou907/DDBM/resolve/main"
    files = [
        (f"{HF_BASE}/edges2handbags_ref_64_data.npz",
         stats_dir / "edges2handbags_ref_64_data.npz"),
        (f"{HF_BASE}/diode_ref_256_data.npz",
         stats_dir / "diode_ref_256_data.npz"),
    ]
    for url, dest in files:
        if dest.exists():
            print(f"[stats] {dest.name} already exists, skipping")
        else:
            print(f"[stats] Downloading {dest.name}...", flush=True)
            run_cmd(f"wget -q {url} -O {dest}")


# ---------------------------------------------------------------------------
# 6. Inception model (for FID evaluator)
# ---------------------------------------------------------------------------
def prepare_inception(base_dir: Path):
    """Download Inception V3 weights for FID evaluation.

    The pre_edit points feature_extractor.py to look for inception at
    /workspace/dbim-codebase/assets/ which maps to base_dir.
    """
    inception_path = base_dir / "inception-2015-12-05.pt"
    if inception_path.exists():
        print("[inception] inception-2015-12-05.pt already exists, skipping")
    else:
        print("[inception] Downloading Inception V3 TorchScript...", flush=True)
        run_cmd(
            "wget -q https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/"
            f"pretrained/metrics/inception-2015-12-05.pt -O {inception_path}"
        )

    pt_inception_path = base_dir / "pt_inception-2015-12-05-6726825d.pth"
    if pt_inception_path.exists():
        print("[inception] pt_inception-2015-12-05-6726825d.pth already exists, skipping")
    else:
        print("[inception] Downloading PyTorch FID Inception weights...", flush=True)
        run_cmd(
            "wget -q https://github.com/mseitzer/pytorch-fid/releases/download/"
            f"fid_weights/pt_inception-2015-12-05-6726825d.pth -O {pt_inception_path}"
        )


# ---------------------------------------------------------------------------
# 7. Torch model cache
# ---------------------------------------------------------------------------
def prepare_torch_cache(base_dir: Path):
    """Download torchvision weights needed by runtime evaluators."""
    checkpoints_dir = base_dir / "torch_cache" / "hub" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    files = [
        (
            "https://download.pytorch.org/models/resnet50-0676ba61.pth",
            checkpoints_dir / "resnet50-0676ba61.pth",
        ),
        (
            "https://download.pytorch.org/models/vgg16-397923af.pth",
            checkpoints_dir / "vgg16-397923af.pth",
        ),
        # photosynthesis.metrics.LPIPS auto-downloads this on first import.
        # Stage it here so KarrasDenoiser(loss_norm='lpips') works offline.
        (
            "https://github.com/photosynthesis-team/photosynthesis.metrics/releases/download/v0.4.0/lpips_weights.pt",
            checkpoints_dir / "lpips_weights.pt",
        ),
    ]
    for url, dest in files:
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[torch_cache] {dest.name} already exists, skipping")
        else:
            print(f"[torch_cache] Downloading {dest.name}...", flush=True)
            run_cmd(f"wget -q {url} -O {dest}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Prepare all data for dbim-codebase tasks"
    )
    parser.add_argument("--data-root", type=str, required=True,
                        help="Root data directory (e.g. vendor/data)")
    parser.add_argument("--skip-imagenet", action="store_true",
                        default=os.environ.get("SKIP_IMAGENET", "").lower() in ("1", "true", "yes"),
                        help="Skip ImageNet val download (~6.7 GB zip from HF mirror). Also honored via SKIP_IMAGENET=1 env.")
    parser.add_argument("--skip-diode", action="store_true",
                        default=os.environ.get("SKIP_DIODE", "").lower() in ("1", "true", "yes"),
                        help="Skip DIODE download (~20 GB). Also honored via SKIP_DIODE=1 env.")
    args = parser.parse_args()

    # Base dir maps to /workspace/dbim-codebase/assets via data_bind
    base_dir = Path(args.data_root) / "dbim_data"
    base_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir = base_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: edges2handbags dataset ===")
    prepare_edges2handbags(datasets_dir)

    if not args.skip_diode:
        print("\n=== Step 2: DIODE dataset ===")
        prepare_diode(datasets_dir)
    else:
        print("\n=== Step 2: DIODE (skipped) ===")

    if not args.skip_imagenet:
        print("\n=== Step 3: ImageNet dataset ===")
        prepare_imagenet(datasets_dir)
    else:
        print("\n=== Step 3: ImageNet (skipped) ===")

    print("\n=== Step 4: Checkpoints ===")
    prepare_checkpoints(base_dir / "ckpts")

    print("\n=== Step 5: FID reference statistics ===")
    prepare_stats(base_dir / "stats")

    print("\n=== Step 6: Inception model ===")
    prepare_inception(base_dir)

    print("\n=== Step 7: Torch model cache ===")
    prepare_torch_cache(base_dir)

    print("\n=== All done ===")
    print(f"Output: {base_dir}/")
    print("Data binds:")
    print("  {data_root}/dbim_data:/workspace/dbim-codebase/assets")
    print("  {data_root}/dbim_data/torch_cache:/data/torch_cache")


if __name__ == "__main__":
    main()
