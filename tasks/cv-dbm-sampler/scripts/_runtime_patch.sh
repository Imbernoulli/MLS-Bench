#!/bin/bash
# Verifier-only runtime patch for the DBIM base package.
#
# The upstream DBIM scripts were written for large multi-GPU jobs. On Mangrove
# H20 pods the default verifier memory limit is tighter, so keep the scientific
# metric unchanged while lowering peak CPU RSS in sample materialization and FID
# statistics. Keep GPU-throughput defaults at upstream values unless an env var
# explicitly overrides them.

export DBIM_SAMPLE_BATCH_SIZE="${DBIM_SAMPLE_BATCH_SIZE:-16}"
export DBIM_FID_BATCH_SIZE="${DBIM_FID_BATCH_SIZE:-1024}"
export DBIM_LPIPS_BATCH_SIZE="${DBIM_LPIPS_BATCH_SIZE:-128}"
export DBIM_IMAGENET_ACCU_BATCH_SIZE="${DBIM_IMAGENET_ACCU_BATCH_SIZE:-256}"
export DBIM_IMAGENET_FID_BATCH_SIZE="${DBIM_IMAGENET_FID_BATCH_SIZE:-256}"
export DBIM_REF_FID_BATCH_SIZE="${DBIM_REF_FID_BATCH_SIZE:-512}"
export DBIM_FID_DATAPARALLEL="${DBIM_FID_DATAPARALLEL:-1}"
export DBIM_SKIP_LPIPS="${DBIM_SKIP_LPIPS:-0}"
export DBIM_SKIP_IS="${DBIM_SKIP_IS:-0}"
export DBIM_DISABLE_SAMPLE_LPIPS="${DBIM_DISABLE_SAMPLE_LPIPS:-0}"
export DBIM_NUM_WORKERS="${DBIM_NUM_WORKERS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

_dbim_patch_lock="${DBIM_RUNTIME_PATCH_LOCK:-.dbim_runtime_patch.lock}"
{
if command -v flock >/dev/null 2>&1; then
    flock 9
fi

python3 - <<'PY'
from pathlib import Path


def write_if_changed(path: Path, text: str) -> None:
    old = path.read_text()
    if old != text:
        path.write_text(text)


sample_sh = Path("scripts/sample.sh")
if sample_sh.exists():
    text = sample_sh.read_text()
    text = text.replace("BS=16\n", 'BS=${DBIM_SAMPLE_BATCH_SIZE:-16}\n', 1)
    old_gpu = (
        "export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7\n"
        'run_args="--nproc_per_node 8 \\\n'
        '          --master_port 29511"\n'
    )
    new_gpu = (
        'case "${CUDA_VISIBLE_DEVICES:-}" in\n'
        '    ""|"all"|"none"|"void"|"-1") ;;\n'
        '    *) NGPU=$(printf "%s\\n" "$CUDA_VISIBLE_DEVICES" | tr "," "\\n" | sed "/^[[:space:]]*$/d" | wc -l) ;;\n'
        "esac\n"
        'if [ -z "${NGPU:-}" ] && command -v nvidia-smi >/dev/null 2>&1; then\n'
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
    if "${DBIM_NUM_WORKERS:+ --num_workers=" not in text:
        text = text.replace(
            ' ${ORDER:+ --order="${ORDER}"}\n',
            ' ${ORDER:+ --order="${ORDER}"} ${DBIM_NUM_WORKERS:+ --num_workers="${DBIM_NUM_WORKERS}"}\n',
            1,
        )
    write_if_changed(sample_sh, text)


sample_py = Path("sample.py")
if sample_py.exists():
    text = sample_py.read_text()
    if "import zipfile\n" not in text:
        text = text.replace("import os\n", "import os\nimport zipfile\n", 1)
    if "from contextlib import suppress\n" not in text:
        text = text.replace("import zipfile\n", "import zipfile\nfrom contextlib import suppress\n", 1)
    text = text.replace(
        "    all_images = []\n"
        "    all_labels = []\n",
        "    sample_writer = None\n"
        "    label_writer = None\n"
        "    sample_tmp_path = None\n"
        "    label_tmp_path = None\n"
        "    sample_shape = None\n"
        "    label_shape = None\n"
        "    write_pos = 0\n"
        "    label_write_pos = 0\n"
        "    final_nfe = None\n",
        1,
    )
    text = text.replace(
        "    args.num_samples = len(dataloader.dataset)\n"
        "    num = 0\n",
        "    args.num_samples = len(dataloader.dataset)\n"
        "\n"
        "    def _zip_npy_to_npz(npy_path, npz_path):\n"
        "        with zipfile.ZipFile(npz_path, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as zf:\n"
        "            zf.write(npy_path, 'arr_0.npy')\n"
        "        with suppress(OSError):\n"
        "            os.remove(npy_path)\n"
        "\n"
        "    num = 0\n",
        1,
    )
    text = text.replace(
        "        all_images.append(gathered_samples.detach().cpu().numpy())\n"
        "        if \"inpaint\" in args.dataset:\n"
        "            all_labels.append(gathered_labels.detach().cpu().numpy())\n",
        "        if dist.get_rank() == 0:\n"
        "            batch_np = gathered_samples.detach().cpu().numpy()\n"
        "            if sample_writer is None:\n"
        "                final_nfe = nfe\n"
        "                sample_shape = (args.num_samples, *batch_np.shape[1:])\n"
        "                sample_tmp_path = os.path.join(sample_dir, f'.samples_arr_0_{os.getpid()}.npy')\n"
        "                sample_writer = np.lib.format.open_memmap(\n"
        "                    sample_tmp_path, mode='w+', dtype=batch_np.dtype, shape=sample_shape\n"
        "                )\n"
        "            take = min(batch_np.shape[0], args.num_samples - write_pos)\n"
        "            if take > 0:\n"
        "                sample_writer[write_pos: write_pos + take] = batch_np[:take]\n"
        "                write_pos += take\n"
        "                sample_writer.flush()\n"
        "            del batch_np\n"
        "\n"
        "            if \"inpaint\" in args.dataset:\n"
        "                label_np = gathered_labels.detach().cpu().numpy()\n"
        "                if label_writer is None:\n"
        "                    label_shape = (args.num_samples, *label_np.shape[1:])\n"
        "                    label_tmp_path = os.path.join(sample_dir, f'.labels_arr_0_{os.getpid()}.npy')\n"
        "                    label_writer = np.lib.format.open_memmap(\n"
        "                        label_tmp_path, mode='w+', dtype=label_np.dtype, shape=label_shape\n"
        "                    )\n"
        "                take = min(label_np.shape[0], args.num_samples - label_write_pos)\n"
        "                if take > 0:\n"
        "                    label_writer[label_write_pos: label_write_pos + take] = label_np[:take]\n"
        "                    label_write_pos += take\n"
        "                    label_writer.flush()\n"
        "                del label_np\n",
        1,
    )
    text = text.replace(
        "    logger.log(f\"created {len(all_images) * args.batch_size * dist.get_world_size()} samples\")\n"
        "\n"
        "    arr = np.concatenate(all_images, axis=0)\n"
        "    arr = arr[: args.num_samples]\n"
        "    if \"inpaint\" in args.dataset:\n"
        "        labels = np.concatenate(all_labels, axis=0)\n"
        "        labels = labels[: args.num_samples]\n"
        "\n"
        "    if dist.get_rank() == 0:\n"
        "        shape_str = \"x\".join([str(x) for x in arr.shape])\n"
        "        out_path = os.path.join(sample_dir, f\"samples_{shape_str}_nfe{nfe}.npz\")\n"
        "        logger.log(f\"saving to {out_path}\")\n"
        "        np.savez(out_path, arr)\n"
        "        if \"inpaint\" in args.dataset:\n"
        "            shape_str = \"x\".join([str(x) for x in labels.shape])\n"
        "            out_path = os.path.join(sample_dir, f\"labels_{shape_str}_nfe{nfe}.npz\")\n"
        "            logger.log(f\"saving to {out_path}\")\n"
        "            np.savez(out_path, labels)\n",
        "    logger.log(f\"created {num} samples\")\n"
        "\n"
        "    if dist.get_rank() == 0:\n"
        "        if sample_writer is None or sample_shape is None or final_nfe is None:\n"
        "            raise RuntimeError('no samples were written')\n"
        "        sample_writer.flush()\n"
        "        del sample_writer\n"
        "        shape_str = \"x\".join([str(x) for x in sample_shape])\n"
        "        out_path = os.path.join(sample_dir, f\"samples_{shape_str}_nfe{final_nfe}.npz\")\n"
        "        logger.log(f\"saving to {out_path}\")\n"
        "        _zip_npy_to_npz(sample_tmp_path, out_path)\n"
        "        if \"inpaint\" in args.dataset:\n"
        "            if label_writer is None or label_shape is None:\n"
        "                raise RuntimeError('no labels were written')\n"
        "            label_writer.flush()\n"
        "            del label_writer\n"
        "            shape_str = \"x\".join([str(x) for x in label_shape])\n"
        "            out_path = os.path.join(sample_dir, f\"labels_{shape_str}_nfe{final_nfe}.npz\")\n"
        "            logger.log(f\"saving to {out_path}\")\n"
        "            _zip_npy_to_npz(label_tmp_path, out_path)\n",
        1,
    )
    write_if_changed(sample_py, text)


evaluate_sh = Path("scripts/evaluate.sh")
if evaluate_sh.exists():
    text = evaluate_sh.read_text()
    text = text.replace(
        "    python evaluations/evaluator.py $REF_PATH $SAMPLE_PATH --metric lpips\n",
        '    if [[ "${DBIM_SKIP_LPIPS:-0}" != "1" ]]; then\n'
        "        python evaluations/evaluator.py $REF_PATH $SAMPLE_PATH --metric lpips\n"
        "    fi\n",
        1,
    )
    text = text.replace(
        '    python evaluations/evaluator.py "" $SAMPLE_PATH --metric is\n',
        '    if [[ "${DBIM_SKIP_IS:-0}" != "1" ]]; then\n'
        '        python evaluations/evaluator.py "" $SAMPLE_PATH --metric is\n'
        "    fi\n",
        1,
    )
    write_if_changed(evaluate_sh, text)


evaluator_py = Path("evaluations/evaluator.py")
if evaluator_py.exists():
    text = evaluator_py.read_text()
    start = text.find("def get_fid(args):\n")
    end = text.find("\ndef get_ssim_lpips(args):\n")
    if start != -1 and end != -1 and "def _stream_fid_statistics(" not in text:
        text = text[:start] + r'''def _stream_fid_statistics(evaluator, npz_path):
    """Compute FID mean/cov online without retaining all activations."""
    n = 0
    sum_act = None
    sum_outer = None
    with open_npz_array(npz_path, "arr_0") as reader:
        for batch in tqdm(reader.read_batches(evaluator.batch_size)):
            if MODE == "legacy_tensorflow":
                batch_t = torch.from_numpy(batch.transpose([0, 3, 1, 2])).float()
            else:
                batch_t = evaluator.resize_batch(batch)
            pred, _ = evaluator.model(batch_t)
            act = pred.detach().cpu().numpy().reshape([pred.shape[0], -1]).astype(np.float64, copy=False)
            if sum_act is None:
                dim = act.shape[1]
                sum_act = np.zeros(dim, dtype=np.float64)
                sum_outer = np.zeros((dim, dim), dtype=np.float64)
            sum_act += act.sum(axis=0)
            sum_outer += act.T @ act
            n += act.shape[0]
            del batch_t, pred, act
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if n < 2:
        raise ValueError(f"need at least two activations for FID, got {n}")
    mu = sum_act / n
    sigma = (sum_outer - n * np.outer(mu, mu)) / (n - 1)
    return FIDStatistics(mu, sigma)


def _cached_or_stream_fid_statistics(evaluator, npz_path):
    obj = np.load(npz_path)
    keys = set(obj.keys())
    if "mu" in keys and "sigma" in keys:
        return FIDStatistics(obj["mu"], obj["sigma"])
    if os.environ.get("DBIM_FID_STREAM_STATS", "1") == "1" and "arr_0" in keys:
        return _stream_fid_statistics(evaluator, npz_path)
    if "pool_3" in keys:
        return evaluator.compute_statistics(obj["pool_3"])
    if "act" in keys:
        return evaluator.compute_statistics(obj["act"])
    return _stream_fid_statistics(evaluator, npz_path)


def get_fid(args):
    model = build_feature_extractor(
        MODE,
        torch.device("cuda"),
        use_dataparallel=os.environ.get("DBIM_FID_DATAPARALLEL", "0") == "1",
    )
    evaluator = Evaluator(model, batch_size=int(os.environ.get("DBIM_FID_BATCH_SIZE", "1024")))

    print("computing/reading reference batch statistics...")
    ref_stats = _cached_or_stream_fid_statistics(evaluator, args.ref_batch)
    print("computing sample batch statistics...")
    sample_stats = _stream_fid_statistics(evaluator, args.sample_batch)

    print("Computing evaluations...")
    fid = sample_stats.frechet_distance(ref_stats)
    metrics = {"fid": fid, "inception": 0.0}
    pprint(metrics)

    save_dir = os.path.dirname(os.path.realpath(args.sample_batch))
    with open(os.path.join(save_dir, "fid.json"), "w") as f:
        json.dump(metrics, f)
''' + text[end + 1:]
    text = text.replace(
        'build_feature_extractor(MODE, torch.device("cuda"), use_dataparallel=True)',
        'build_feature_extractor(MODE, torch.device("cuda"), '
        'use_dataparallel=os.environ.get("DBIM_FID_DATAPARALLEL", "0") == "1")',
    )
    text = text.replace(
        "Evaluator(model, batch_size=1024)",
        'Evaluator(model, batch_size=int(os.environ.get("DBIM_FID_BATCH_SIZE", "1024")))',
    )
    text = text.replace(
        "batch_size=128, shuffle=False, num_workers=1",
        'batch_size=int(os.environ.get("DBIM_LPIPS_BATCH_SIZE", "128")), '
        "shuffle=False, num_workers=1",
        1,
    )
    write_if_changed(evaluator_py, text)


imagenet_py = Path("evaluation/compute_metrices_imagenet.py")
if imagenet_py.exists():
    text = imagenet_py.read_text()
    if "import atexit\n" not in text:
        text = text.replace(
            "import random\n",
            "import random\n"
            "import atexit\n"
            "import shutil\n"
            "import tempfile\n"
            "import zipfile\n",
            1,
        )
    if "def load_npz_array_mmap(" not in text:
        text = text.replace(
            "ADM_IMG256_FID_TRAIN_REF_CKPT = \"https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz\"\n",
            "ADM_IMG256_FID_TRAIN_REF_CKPT = \"https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz\"\n"
            "_MMAP_TMP_DIRS = []\n"
            "\n"
            "def _cleanup_mmap_tmp_dirs():\n"
            "    for tmp_dir in _MMAP_TMP_DIRS:\n"
            "        shutil.rmtree(tmp_dir, ignore_errors=True)\n"
            "\n"
            "atexit.register(_cleanup_mmap_tmp_dirs)\n"
            "\n"
            "def load_npz_array_mmap(path, arr_name='arr_0'):\n"
            "    path = Path(path)\n"
            "    if path.suffix != '.npz':\n"
            "        return np.load(path, mmap_mode='r')\n"
            "    tmp_dir = Path(tempfile.mkdtemp(prefix='dbim-npz-mmap-'))\n"
            "    _MMAP_TMP_DIRS.append(tmp_dir)\n"
            "    member = f'{arr_name}.npy'\n"
            "    out_path = tmp_dir / member\n"
            "    with zipfile.ZipFile(path, 'r') as zf:\n"
            "        with zf.open(member, 'r') as src, out_path.open('wb') as dst:\n"
            "            shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)\n"
            "    return np.load(out_path, mmap_mode='r')\n",
            1,
        )
    text = text.replace(
        "        batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=1, drop_last=False,\n",
        "        batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=0, drop_last=False,\n",
        1,
    )
    text = text.replace(
        "    numpy_arr = np.load(ckpt_path)['arr_0']\n"
        "    label_path = opt.label\n"
        "    label_arr = np.load(label_path)['arr_0']\n",
        "    numpy_arr = load_npz_array_mmap(ckpt_path)\n"
        "    label_path = opt.label\n"
        "    label_arr = load_npz_array_mmap(label_path)\n",
        1,
    )
    text = text.replace(
        "accu = compute_accu(opt, numpy_arr, numpy_label_arr)",
        'accu = compute_accu(opt, numpy_arr, numpy_label_arr, '
        'batch_size=int(os.environ.get("DBIM_IMAGENET_ACCU_BATCH_SIZE", "256")))',
        1,
    )
    text = text.replace(
        "fid = fid_util.compute_fid_from_numpy(numpy_arr, ref_mu, ref_sigma, mode=opt.mode)",
        'fid = fid_util.compute_fid_from_numpy('
        'numpy_arr, ref_mu, ref_sigma, '
        'batch_size=int(os.environ.get("DBIM_IMAGENET_FID_BATCH_SIZE", "256")), '
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
        'dataset, mode, batch_size=int(os.environ.get("DBIM_REF_FID_BATCH_SIZE", "512")), '
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
        '__import__("os").environ.get("DBIM_DISABLE_SAMPLE_LPIPS", "0") != "1":\n'
        '            self.lpips_loss = LPIPS(replace_pooling=True, reduction="none")',
        1,
    )
    write_if_changed(karras_py, text)
PY
} 9>"${_dbim_patch_lock}"
