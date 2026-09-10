#!/bin/bash

SEED=${SEED:-42}

# Read per-method overrides from CONFIG_OVERRIDES dict in models/Custom.py.
# Allowed keys: n_hidden (int), slice_num (int).
read_override() {
  python -c "import importlib.util, sys
spec = importlib.util.spec_from_file_location('cm', 'models/Custom.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
ov = getattr(m, 'CONFIG_OVERRIDES', {}) or {}
v = ov.get('$1', None)
print(v if v is not None else '')" 2>/dev/null
}

N_HIDDEN=$(read_override n_hidden)
SLICE_NUM=$(read_override slice_num)
N_HIDDEN=${N_HIDDEN:-128}
SLICE_NUM=${SLICE_NUM:-32}

python run.py \
  --gpu 0 \
  --data_path /data/AirCraft/ \
  --loader aircraft_design \
  --geotype unstructured \
  --task steady_design \
  --space_dim 3 \
  --fun_dim 7 \
  --out_dim 6 \
  --model Custom \
  --n_hidden $N_HIDDEN \
  --n_heads 8 \
  --n_layers 8 \
  --mlp_ratio 2 \
  --slice_num $SLICE_NUM \
  --unified_pos 0 \
  --ref 8 \
  --batch-size 1 \
  --epochs 200 \
  --eval 0 \
  --max_grad_norm 1.0 \
  --seed $SEED \
  --save_name aircraft_Custom_s${SEED}
