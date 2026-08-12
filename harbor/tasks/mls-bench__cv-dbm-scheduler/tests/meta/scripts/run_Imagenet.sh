# Seed the piq FID cache with a pre-shipped copy of pt_inception weights.
# Compute nodes have no network so torch.hub can't download it on first FID call.
# The .pth ships under assets/ (the data mount) and we stage it to TORCH_HOME.
set -euo pipefail

if [ -f assets/pt_inception-2015-12-05-6726825d.pth ]; then
    mkdir -p "${TORCH_HOME:-/data/torch_cache}/hub/checkpoints"
    cp -n assets/pt_inception-2015-12-05-6726825d.pth \
          "${TORCH_HOME:-/data/torch_cache}/hub/checkpoints/" 2>/dev/null || true
fi

# Inject --num_samples into sample.py + make the ImageNet LMDB eval fork-safe
# (evaluator.py _ForkSafeLmdb) so the hidden ImageNet setting actually produces
# samples_*.npz instead of dying on a concurrent-LMDB race.
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
# (1) DATA: the HF mirror val.zip was packed on macOS, so it carries __MACOSX/
#     AppleDouble cruft (a 212-byte ._<image>.JPEG sidecar per real image).
#     ImageFolder otherwise picks up the ._* (they end in .JPEG) -> the FID ref
#     reads a sidecar and dies UnidentifiedImageError -> task scores 0. Delete
#     the cruft to recover the canonical clean 50k val.
# (2) RACE: the pt+lmdb cache is built lazily on first access; 4 torchrun ranks
#     racing to build the same cache truncate/corrupt it. Build once
#     single-process here so ranks only load it.
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

# Drop this dataset's res.json left by a previous iteration so a failed run
# cannot re-echo stale metrics. Scoped to the ImageNet model dirs: other
# settings may be mid-eval in the shared workdir/.
find workdir/ -type f -path "*/imagenet256_inpaint_ema_*" -name "res.json" -delete 2>/dev/null || true

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

# Clean up THIS dataset's sample/label NPZs once FID has been computed — each
# agent iteration would otherwise keep a ~2 GB (10k × 256x256x3 uint8) NPZ on
# Vepfs. The delete is scoped to the ImageNet model dirs: all three settings
# share workdir/ and can run concurrently, so an unscoped `find workdir/
# -delete` races a concurrent setting that is still reading its own samples
# and zeroes its FID (issue #79). -print leaves an audit trail in the eval log.
find workdir/ -type f -path "*/imagenet256_inpaint_ema_*" \
    \( -name "samples_*.npz" -o -name "labels_*.npz" \) -print -delete 2>/dev/null || true
rm -rf "$sample_dir"
