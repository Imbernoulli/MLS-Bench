export eta=0.0
export ds=e2h
export num_samples=10000
export doob_scale=1.0
export sampler=dbim
export nfe=5

export sample_dir=${OUTPUT_DIR:-output}/$ds-$nfe-$sampler-$eta-seed${SEED:-42}
rm -rf "$sample_dir"
mkdir -p "$sample_dir"
bash scripts/sample.sh $ds $nfe $sampler $eta
bash scripts/evaluate.sh $ds $nfe $sampler $eta
if [ -f "$sample_dir/fid.json" ]; then
    echo "FID: $(python3 -c "import json; print(json.load(open(\"$sample_dir/fid.json\"))['fid'])")"
fi

# Clean up THIS dataset's sample NPZs once FID has been computed — each agent
# iteration would otherwise keep a ~60 MB (10k × 64x64x3 uint8) NPZ on Vepfs.
# The delete is scoped to the e2h model dirs: all three settings share
# workdir/ and can run concurrently, so an unscoped `find workdir/ -delete`
# races a concurrent setting that is still reading its own samples and zeroes
# its FID (issue #79). -print leaves an audit trail in the eval log.
find workdir/ -type f -path "*/e2h_ema_*" -name "samples_*.npz" -print -delete 2>/dev/null || true
rm -rf "$sample_dir"
