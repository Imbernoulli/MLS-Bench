set -euo pipefail

export eta=0.0
export ds=diode
export num_samples=10000
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

find workdir/ -name "samples_*.npz" -delete 2>/dev/null || true
rm -rf "$sample_dir"
