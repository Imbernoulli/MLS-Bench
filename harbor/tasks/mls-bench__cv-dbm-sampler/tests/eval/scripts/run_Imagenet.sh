# Seed the piq FID cache with a pre-shipped copy of pt_inception weights.
# Compute nodes have no network so torch.hub can't download it on first FID call.
# The .pth ships under assets/ (the data mount) and we stage it to TORCH_HOME.
if [ -f assets/pt_inception-2015-12-05-6726825d.pth ]; then
    mkdir -p "${TORCH_HOME:-/data/torch_cache}/hub/checkpoints"
    cp -n assets/pt_inception-2015-12-05-6726825d.pth \
          "${TORCH_HOME:-/data/torch_cache}/hub/checkpoints/" 2>/dev/null || true
fi

source "$(dirname "${BASH_SOURCE[0]}")/_runtime_patch.sh"

export eta=0.0
export ds=imagenet_inpaint_center
export num_samples=10000
export doob_scale=1.0
export sampler=dbim
export nfe=5

export sample_dir=${OUTPUT_DIR:-output}/$ds-$nfe-$sampler-$eta-seed${SEED:-42}
rm -rf "$sample_dir"
mkdir -p "$sample_dir"

# Two ImageNet-only data/runtime fixes, both single-process before the 4-rank
# torchrun. Neither changes sampling/model/FID/scientific setup.
#
# (1) DATA: the HF mirror mlx-vision/imagenet-1k val.zip was packed on macOS, so
#     it carries __MACOSX/ AppleDouble cruft (a 212-byte ._<image>.JPEG sidecar
#     per real image). prepare_data.py doesn't filter it, so val holds ~100k
#     entries (50k real + 50k ._* non-images) and ImageFolder picks up the ._*
#     (they end in .JPEG). The FID reference is generated from the FULL val on
#     first run, so it reads a ._* sidecar and dies UnidentifiedImageError ->
#     best_fid_Imagenet missing -> task scores 0. Delete the cruft to recover the
#     canonical clean 50k val (so the FID ref is correct and comparable to the
#     leaderboard baselines). This restores the intended data; it does not alter
#     the eval.
#
# (2) RACE: pt+lmdb are built lazily on first access; 4 torchrun ranks racing to
#     build the same cache truncate/corrupt it (EOFError / UnidentifiedImageError
#     / key miss / "already open"). Build once single-process here so ranks only
#     load it.
export PYTHONPATH="${PYTHONPATH:-}:."
python3 - <<'PYWARM' || echo "[warmup] non-fatal failure (see diagnostic above)"
import os, os.path as osp, shutil, sys, glob
sys.path.insert(0, ".")
DATA_DIR = "assets/datasets/ImageNet"
root = osp.join(DATA_DIR, "val")
pt = root + "_faster_imagefolder.lmdb.pt"
lm = root + "_faster_imagefolder.lmdb"

# (1) Strip macOS AppleDouble cruft so ImageFolder sees only the real 50k images.
mac = osp.join(root, "__MACOSX")
if osp.isdir(mac):
    shutil.rmtree(mac, ignore_errors=True)
    print("[warmup] removed __MACOSX/ dir")
_apdbl = 0
for p in glob.iglob(osp.join(root, "**", "._*"), recursive=True):
    try:
        os.remove(p)
        _apdbl += 1
    except Exception:
        pass
print("[warmup] removed %d stray ._* AppleDouble files" % _apdbl)
try:
    nreal = sum(1 for _ in glob.iglob(osp.join(root, "*", "*.JPEG")))
    print("[warmup] real val/<synset>/*.JPEG after cleanup:", nreal, "(expect ~50000)")
except Exception as e:
    print("[warmup] count err", repr(e)[:120])

# (2) Force a clean single-process rebuild from the cleaned val (the cache, if
#     any, was built from the dirty 100k val).
if osp.isfile(pt):
    try:
        os.remove(pt)
    except Exception:
        pass
if osp.isdir(lm):
    shutil.rmtree(lm, ignore_errors=True)
print("[warmup] building clean cache single-process ...")
from datasets.imagenet_inpaint import build_lmdb_dataset_val10k
ds = build_lmdb_dataset_val10k(DATA_DIR, 256)
print("[warmup] clean cache ready; val10k len =", len(ds))
PYWARM
bash scripts/sample.sh $ds $nfe $sampler $eta
bash scripts/evaluate.sh $ds $nfe $sampler $eta

# compute_metrices_imagenet.py writes res.json (not fid.json) with both
# accuracy and FID. Surface them as "FID: <num>" so the task parser picks
# up the score via its first regex.
RES_JSON=$(ls workdir/imagenet256_inpaint_ema_*/sample_*/split=test/*/steps=*/res.json 2>/dev/null | head -1)
if [ -n "$RES_JSON" ]; then
    echo "FID: $(python3 -c "import json; print(json.load(open('$RES_JSON'))['fid'])")"
    echo "Accuracy: $(python3 -c "import json; print(json.load(open('$RES_JSON'))['accu'])")"
fi

# Clean up sample NPZ files once FID has been computed — each agent iteration
# would otherwise keep a ~2 GB (10k × 256x256x3 uint8) NPZ on Vepfs.
find workdir/ -name "samples_*.npz" -delete 2>/dev/null || true
find workdir/ -name "labels_*.npz" -delete 2>/dev/null || true
rm -rf "$sample_dir"
