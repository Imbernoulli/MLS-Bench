export eta=0.0
export ds=diode
export num_samples=10000
export doob_scale=1.0
export sampler=dbim
export nfe=50

export sample_dir=${OUTPUT_DIR:-output}/$ds-$nfe-$sampler-$eta-seed${SEED:-42}
rm -rf "$sample_dir"
mkdir -p "$sample_dir"
bash scripts/sample.sh $ds $nfe $sampler $eta
bash scripts/evaluate.sh $ds $nfe $sampler $eta
if [ -f "$sample_dir/fid.json" ]; then
    echo "FID: $(python3 -c "import json; print(json.load(open(\"$sample_dir/fid.json\"))['fid'])")"
fi

find workdir/ -name "samples_*.npz" -delete 2>/dev/null || true
rm -rf "$sample_dir"
