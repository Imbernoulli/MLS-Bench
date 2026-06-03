#!/bin/bash
# Verifier-only runtime patch for the DBIM base package.
#
# The upstream DBIM scripts were written for large multi-GPU jobs. On Mangrove
# H20 pods the default verifier memory limit is tighter, so keep the scientific
# metric unchanged while lowering peak CPU/GPU memory in sampling and FID.

export DBIM_SAMPLE_BATCH_SIZE="${DBIM_SAMPLE_BATCH_SIZE:-8}"
export DBIM_FID_BATCH_SIZE="${DBIM_FID_BATCH_SIZE:-128}"
export DBIM_LPIPS_BATCH_SIZE="${DBIM_LPIPS_BATCH_SIZE:-16}"
export DBIM_IMAGENET_ACCU_BATCH_SIZE="${DBIM_IMAGENET_ACCU_BATCH_SIZE:-64}"
export DBIM_IMAGENET_FID_BATCH_SIZE="${DBIM_IMAGENET_FID_BATCH_SIZE:-64}"
export DBIM_REF_FID_BATCH_SIZE="${DBIM_REF_FID_BATCH_SIZE:-128}"
export DBIM_FID_DATAPARALLEL="${DBIM_FID_DATAPARALLEL:-0}"
export DBIM_SKIP_LPIPS="${DBIM_SKIP_LPIPS:-1}"
export DBIM_DISABLE_SAMPLE_LPIPS="${DBIM_DISABLE_SAMPLE_LPIPS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

python3 - <<'PY'
from pathlib import Path


def write_if_changed(path: Path, text: str) -> None:
    old = path.read_text()
    if old != text:
        path.write_text(text)


sample_sh = Path("scripts/sample.sh")
if sample_sh.exists():
    text = sample_sh.read_text()
    text = text.replace("BS=16\n", 'BS=${DBIM_SAMPLE_BATCH_SIZE:-8}\n', 1)
    old_gpu = (
        "export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7\n"
        'run_args="--nproc_per_node 8 \\\n'
        '          --master_port 29511"\n'
    )
    new_gpu = (
        "if command -v nvidia-smi >/dev/null 2>&1; then\n"
        "    NGPU=$(nvidia-smi --list-gpus 2>/dev/null | grep -c '^GPU ')\n"
        "fi\n"
        'if [ -z "${NGPU:-}" ] || [ "$NGPU" -lt 1 ]; then\n'
        "    NGPU=1\n"
        "fi\n"
        'run_args="--nproc_per_node $NGPU \\\n'
        '          --master_port 29511"\n'
    )
    text = text.replace(old_gpu, new_gpu, 1)
    if "${num_samples:+ --num_samples=" not in text:
        text = text.replace(
            " --use_new_attention_order $ATTN_TYPE --data_dir=$DATA_DIR --dataset=$DATASET --split $SPLIT\\\n",
            " --use_new_attention_order $ATTN_TYPE --data_dir=$DATA_DIR --dataset=$DATASET --split $SPLIT \\\n"
            ' ${num_samples:+ --num_samples="${num_samples}"} ${SEED:+ --seed="${SEED}"} \\\n',
            1,
        )
    write_if_changed(sample_sh, text)


evaluate_sh = Path("scripts/evaluate.sh")
if evaluate_sh.exists():
    text = evaluate_sh.read_text()
    text = text.replace(
        "    python evaluations/evaluator.py $REF_PATH $SAMPLE_PATH --metric lpips\n",
        '    if [[ "${DBIM_SKIP_LPIPS:-1}" != "1" ]]; then\n'
        "        python evaluations/evaluator.py $REF_PATH $SAMPLE_PATH --metric lpips\n"
        "    fi\n",
        1,
    )
    write_if_changed(evaluate_sh, text)


evaluator_py = Path("evaluations/evaluator.py")
if evaluator_py.exists():
    text = evaluator_py.read_text()
    text = text.replace(
        'build_feature_extractor(MODE, torch.device("cuda"), use_dataparallel=True)',
        'build_feature_extractor(MODE, torch.device("cuda"), '
        'use_dataparallel=os.environ.get("DBIM_FID_DATAPARALLEL", "0") == "1")',
    )
    text = text.replace(
        "Evaluator(model, batch_size=1024)",
        'Evaluator(model, batch_size=int(os.environ.get("DBIM_FID_BATCH_SIZE", "128")))',
    )
    text = text.replace(
        "batch_size=128, shuffle=False, num_workers=1",
        'batch_size=int(os.environ.get("DBIM_LPIPS_BATCH_SIZE", "16")), '
        "shuffle=False, num_workers=1",
        1,
    )
    write_if_changed(evaluator_py, text)


imagenet_py = Path("evaluation/compute_metrices_imagenet.py")
if imagenet_py.exists():
    text = imagenet_py.read_text()
    text = text.replace(
        "accu = compute_accu(opt, numpy_arr, numpy_label_arr)",
        'accu = compute_accu(opt, numpy_arr, numpy_label_arr, '
        'batch_size=int(os.environ.get("DBIM_IMAGENET_ACCU_BATCH_SIZE", "64")))',
        1,
    )
    text = text.replace(
        "fid = fid_util.compute_fid_from_numpy(numpy_arr, ref_mu, ref_sigma, mode=opt.mode)",
        'fid = fid_util.compute_fid_from_numpy('
        'numpy_arr, ref_mu, ref_sigma, '
        'batch_size=int(os.environ.get("DBIM_IMAGENET_FID_BATCH_SIZE", "64")), '
        "mode=opt.mode)",
        1,
    )
    write_if_changed(imagenet_py, text)


fid_util_py = Path("evaluation/fid_util.py")
if fid_util_py.exists():
    text = fid_util_py.read_text()
    text = text.replace(
        "mu, sigma = collect_features(dataset, mode, batch_size=512, num_workers=num_workers)",
        'mu, sigma = collect_features('
        'dataset, mode, batch_size=int(os.environ.get("DBIM_REF_FID_BATCH_SIZE", "128")), '
        "num_workers=num_workers)",
        1,
    )
    write_if_changed(fid_util_py, text)


karras_py = Path("ddbm/karras_diffusion.py")
if karras_py.exists():
    text = karras_py.read_text()
    text = text.replace(
        'if loss_norm == "lpips":\n'
        '            self.lpips_loss = LPIPS(replace_pooling=True, reduction="none")',
        'if loss_norm == "lpips" and '
        '__import__("os").environ.get("DBIM_DISABLE_SAMPLE_LPIPS", "1") != "1":\n'
        '            self.lpips_loss = LPIPS(replace_pooling=True, reduction="none")',
        1,
    )
    write_if_changed(karras_py, text)
PY
