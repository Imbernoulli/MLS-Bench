set -euo pipefail

export eta=0.0
export ds=diode
# Evaluate the FULL DIODE split (16502 images), not a 10000-image prefix.
#
# The prefix was irreproducible. `datasets/aligned_dataset.py` builds DIODE's
# file list with a bare `os.listdir()` and never sorts it (the sibling
# EdgesDataset does sort, at L92), so "the first 10000" was whatever order the
# filesystem happened to return. Measured on cv-dbm-sampler's `dbim` baseline --
# same code, weights, seed and hardware, only the listing order differing:
#
#     num_samples   natural (ext4) order   sorted order
#         10000                 15.0316        30.2292   <- 101% swing
#         16502                 14.2039        14.2469   <- 0.3%
#
# Sorting groups the filenames by scene, so a sorted 10000-prefix truncated the
# eval from 97 scenes to 61 and shared only 61% of its images with the natural
# order. The subset also shifted with the rank count, because DistributedSampler
# interleaves indices across ranks before the early exit.
#
# Sampling the whole split removes the dependency: FID is a statistic over a
# set, and the set is now the entire split whatever order it is listed in. The
# 0.043 residual above is seed noise rather than order -- sample.py seeds each
# image as `indexes + seed`, so reordering the list relabels every noise draw;
# three seeds at natural order span 0.0393 (sd 0.0210), putting the sorted run
# 1.6 sd out, against 723 sd before.
#
# It is also the comparison the reference was built for:
# assets/stats/diode_ref_256_data.npz holds exactly these 16502 images and
# upstream's evaluate.sh expects samples_16502x256x256x3. sample.py clamps
# num_samples to len(dataset), so this stays correct if the split ever grows.
export num_samples=16502
export doob_scale=1.0
export sampler=dbim
export nfe=3

# Inject --num_samples into sample.py so exactly $num_samples images are
# generated and the eval ref/sample batch shapes match.
source "$(dirname "${BASH_SOURCE[0]}")/_runtime_patch.sh"

export sample_dir=${OUTPUT_DIR:-output}/$ds-$nfe-$sampler-$eta-seed${SEED:-42}
rm -rf "$sample_dir"
mkdir -p "$sample_dir"
bash scripts/sample.sh $ds $nfe $sampler $eta
# FID computed first (streaming get_fid); tolerate the paired-LPIPS crash that
# follows on N-mismatch and surface FID from wherever fid.json landed.
# Scope fid.json to THIS dataset's trees ("*${ds}*" matches the model-prefix
# dirs e2h_*/diode_* and $sample_dir): all three settings share workdir/ and
# run concurrently in the same group, so an unscoped `find | head -1` could
# surface another dataset's fid.json (DIODE echoing edges2handbags' FID).
# Pre-delete this dataset's stale fid.json so a crashed eval behind `|| true`
# can't re-echo a previous iteration's value.
find workdir "$sample_dir" "${OUTPUT_DIR:-output}" -ipath "*${ds}*" -name fid.json -delete 2>/dev/null || true
bash scripts/evaluate.sh $ds $nfe $sampler $eta || true
FID_JSON=$(find workdir "$sample_dir" "${OUTPUT_DIR:-output}" -ipath "*${ds}*" -name fid.json 2>/dev/null | head -1)
if [ -n "$FID_JSON" ]; then
    echo "FID: $(python3 -c "import json; print(json.load(open('$FID_JSON'))['fid'])")"
fi

# Scoped to the diode model dirs: all three settings share workdir/ and can
# run concurrently, so an unscoped `find workdir/ -delete` races a concurrent
# setting that is still reading its own samples and zeroes its FID (issue #79).
find workdir/ -type f -path "*/diode_ema_*" -name "samples_*.npz" -print -delete 2>/dev/null || true
rm -rf "$sample_dir"
