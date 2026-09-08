set -euo pipefail

export eta=0.0
export ds=e2h
export num_samples=10000
export doob_scale=1.0
export sampler=dbim
export nfe=5

# Inject --num_samples into sample.py so exactly $num_samples images are
# generated and the eval ref/sample batch shapes match (without this the base
# image's sample.py emits a mismatched count -> evaluator.py assert fails).
source "$(dirname "${BASH_SOURCE[0]}")/_runtime_patch.sh"

export sample_dir=${OUTPUT_DIR:-output}/$ds-$nfe-$sampler-$eta-seed${SEED:-42}
rm -rf "$sample_dir"
mkdir -p "$sample_dir"
bash scripts/sample.sh $ds $nfe $sampler $eta
# FID is computed first by the patched (streaming) get_fid and written next to
# the samples npz. The subsequent paired SSIM/LPIPS metric asserts ref/sample
# have equal N and core-dumps when they differ (it does here). FID is the only
# scored metric, so tolerate the LPIPS crash and surface FID from wherever
# get_fid wrote fid.json.
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

# Clean up THIS dataset's sample NPZs once FID has been computed — each agent
# iteration would otherwise keep a ~60 MB (10k × 64x64x3 uint8) NPZ on Vepfs.
# The delete is scoped to the e2h model dirs: all three settings share
# workdir/ and can run concurrently, so an unscoped `find workdir/ -delete`
# races a concurrent setting that is still reading its own samples and zeroes
# its FID (issue #79). -print leaves an audit trail in the eval log.
find workdir/ -type f -path "*/e2h_ema_*" -name "samples_*.npz" -print -delete 2>/dev/null || true
rm -rf "$sample_dir"
