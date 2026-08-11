# Seed the piq FID cache with a pre-shipped copy of pt_inception weights.
# Compute nodes have no network so torch.hub can't download it on first FID call.
# The .pth ships under assets/ (the data mount) and we stage it to TORCH_HOME.
if [ -f assets/pt_inception-2015-12-05-6726825d.pth ]; then
    mkdir -p "${TORCH_HOME:-/data/torch_cache}/hub/checkpoints"
    cp -n assets/pt_inception-2015-12-05-6726825d.pth \
          "${TORCH_HOME:-/data/torch_cache}/hub/checkpoints/" 2>/dev/null || true
fi

export eta=0.0
export ds=imagenet_inpaint_center
export num_samples=10000
export doob_scale=1.0
export sampler=dbim
export nfe=5

export sample_dir=${OUTPUT_DIR:-output}/$ds-$nfe-$sampler-$eta-seed${SEED:-42}
rm -rf "$sample_dir"
mkdir -p "$sample_dir"

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
