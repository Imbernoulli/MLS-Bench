#!/bin/bash
# GPU validation for the image-deshadow package. Prepares data (3 REAL cast-shadow severity
# settings: light/medium/heavy, terciles of MEASURED shadow attenuation on real ISTD (Wang et
# al. CVPR 2018) shadow/shadow-free/mask photo triplets), then runs the three network
# baselines (copy / unet_nomask / unet_mask) at full 400 iters over seeds 42 and 1 for EVERY
# setting, plus cheat probes, printing every DESHADOW_METRICS line. Expected (SP+M-Net
# intuition): copy = do-nothing floor; unet_nomask (blind) beats the floor; unet_mask
# (mask-guided) beats unet_nomask on EVERY setting -- and the ordering holds across all 3
# severities. Every cheat probe scores below the shadowed-input floor. Run on k1 via mlaunch.
#
# NOTE (post real-data swap): this script prepares data from a PRE-STAGED local copy of the
# ISTD HF parquet mirror (ISTD_PARQUET_DIR) since k1 has no general internet access; stage the
# 6 parquet shards (see vendor/data_scripts/image-deshadow/prepare_data.py TRAIN_FILES /
# TEST_FILES) under moonfs first (e.g. rsync'd from the b0-side download at
# /mnt/moonfs/lvbohan-b0/deshadow-real/istd_raw/data/) and point ISTD_PARQUET_DIR at that dir.
set -u
echo "=== worker: proxy + deps ==="
export http_proxy=http://proxy.msh.work:3128 https_proxy=http://proxy.msh.work:3128
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
pip install --no-cache-dir -q 'numpy<2' pillow pyarrow huggingface_hub 2>&1 | tail -1 || true
unset http_proxy https_proxy   # data prep uses the pre-staged parquet dir, no net needed

PKG=/onboarding/image-deshadow/pkg
PREP=/onboarding/image-deshadow/prepare_data.py
DATA_ROOT=/mnt/moonfs/lvbohan-ksyun/deshadow-data
export ISTD_PARQUET_DIR=${ISTD_PARQUET_DIR:-/mnt/moonfs/lvbohan-ksyun/deshadow-data/_raw_istd_parquet}
export DESHADOW_DATA_ROOT=$DATA_ROOT/image-deshadow

echo "=== prepare data (3 REAL-ISTD severity settings) ==="
python "$PREP" --data-root "$DATA_ROOT" 2>&1 | grep -E "ready|floor|triplets|bounds"

cd "$PKG"
mkdir -p /tmp/sol
mk() { echo "$2" > /tmp/sol/$1.py; }
mk net_copy   'def get_network_config(): return {"arch":"copy"}'
mk net_nomask 'def get_network_config(): return {"arch":"unet_nomask"}'
mk net_mask   'def get_network_config(): return {"arch":"unet_mask"}'

run() { # setting solutionfile seed label
  echo "----- $4 setting=$1 (seed $3) -----"
  python harness.py --data-root "$DESHADOW_DATA_ROOT/$1" --surface network \
    --label "$1" --solution "$2" --iters 400 --seed "$3" 2>&1 \
    | grep -E "DESHADOW_METRICS|NETWORK_APPLIED|NETWORK_FALLBACK|DEVICE"
}

for st in light medium heavy; do
  for s in 42 1; do
    echo "############ SETTING $st seed $s ############"
    run "$st" /tmp/sol/net_copy.py   "$s" "network=copy (do-nothing floor)"
    run "$st" /tmp/sol/net_nomask.py "$s" "network=unet_nomask (blind)"
    run "$st" /tmp/sol/net_mask.py   "$s" "network=unet_mask (mask-guided SOTA)"
  done
done

echo "############ CHEAT PROBES on 'heavy' (should score below the shadowed-input floor) ############"
python - <<'PY'
import numpy as np, torch, os
d=os.environ["DESHADOW_DATA_ROOT"]+"/heavy"
va=np.load(d+"/val.npz")
shad=torch.tensor(va["shad"]); clean=torch.tensor(va["clean"]); mask=torch.tensor(va["mask"])
def psnr_m(p, thr=0.05):
    p=p.clamp(0,1); sel=(mask>thr).float().expand_as(p)
    se=((p-clean)**2)*sel; den=sel.reshape(sel.shape[0],-1).sum(1).clamp_min(1.0)
    mse=(se.reshape(se.shape[0],-1).sum(1)/den).clamp_min(1e-10)
    return float((10*torch.log10(1/mse)).mean())
print("CHEAT shadowed_input(copy) shadow_psnr=%.4f"%psnr_m(shad))
print("CHEAT all_white            shadow_psnr=%.4f"%psnr_m(torch.ones_like(shad)))
print("CHEAT all_black            shadow_psnr=%.4f"%psnr_m(torch.zeros_like(shad)))
print("CHEAT const_gray0.5        shadow_psnr=%.4f"%psnr_m(torch.full_like(shad,0.5)))
print("CHEAT per_image_mean       shadow_psnr=%.4f"%psnr_m(shad.mean(dim=(1,2,3),keepdim=True).expand_as(shad)))
PY
echo "=== DONE ==="
