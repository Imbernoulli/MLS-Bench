# Stage the FID inception weights and apply the shared runtime patch (streaming
# FID + device-safe), matching run_Imagenet.sh. Without this, DIODE FID only gets
# the patch if an ImageNet run happened to patch the shared evaluator.py first.
if [ -f assets/pt_inception-2015-12-05-6726825d.pth ]; then
    mkdir -p "${TORCH_HOME:-/data/torch_cache}/hub/checkpoints"
    cp -n assets/pt_inception-2015-12-05-6726825d.pth \
          "${TORCH_HOME:-/data/torch_cache}/hub/checkpoints/" 2>/dev/null || true
fi
source "$(dirname "${BASH_SOURCE[0]}")/_runtime_patch.sh"

export eta=0.0
export ds=diode
export num_samples=10000
export doob_scale=1.0
export sampler=dbim
export nfe=5

export sample_dir=${OUTPUT_DIR:-output}/$ds-$nfe-$sampler-$eta-seed${SEED:-42}
rm -rf "$sample_dir"
mkdir -p "$sample_dir"
status=0
bash scripts/sample.sh $ds $nfe $sampler $eta || status=$?
if [ "$status" -ne 0 ]; then
    echo "ERROR: sample.sh exited with status $status for ds=$ds" >&2
else
    # evaluate.sh's exit code alone is not authoritative for success: trailing
    # non-scored metrics can fail after fid.json is written. Success is decided
    # below by the presence of the scored artifact.
    bash scripts/evaluate.sh $ds $nfe $sampler $eta \
        || echo "WARNING: evaluate.sh exited with status $? for ds=$ds" >&2
fi
if [ -f "$sample_dir/fid.json" ]; then
    echo "FID: $(python3 -c "import json; print(json.load(open(\"$sample_dir/fid.json\"))['fid'])")"
elif [ "$status" -eq 0 ]; then
    echo "ERROR: no fid.json produced for ds=$ds" >&2
    status=1
fi

# Clean up THIS dataset's sample NPZs once FID has been computed — each agent
# iteration would otherwise keep a ~2 GB (10k × 256x256x3 uint8) NPZ on Vepfs.
# The delete is scoped to the diode model dirs: all three settings share
# workdir/ and can run concurrently, so an unscoped `find workdir/ -delete`
# races a concurrent setting that is still reading its own samples and zeroes
# its FID (issue #79). -print leaves an audit trail in the eval log.
find workdir/ -type f -path "*/diode_ema_*" -name "samples_*.npz" -print -delete 2>/dev/null || true
rm -rf "$sample_dir"
exit $status
