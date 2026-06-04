"""Pre-edit for dbim-codebase.

Applied to a fresh copy of the package before any task/baseline edits.
Keeps the upstream repo unmodified (vendor/external_packages/ is read-only).
"""

# Replacement for datasets/imagenet_inpaint.py lines 24-67 (the two functions
# `lmdb_loader` and `_build_lmdb_dataset`). Fixes three coupled problems with
# the upstream ImageNet LMDB pipeline:
#
#   * (1b) Key/path mismatch -> every txn.get() returns None -> the whole
#     Imagenet eval crashes (`cannot convert 'NoneType' object to bytes`) and
#     the gmean score collapses to 0. Upstream keys the LMDB by the *absolute
#     build-time path* (datasets.ImageFolder(root).imgs), but queries it with
#     the relative runtime path (assets/datasets/ImageNet/val/<synset>/<file>).
#     The two only match by luck; a snapshot built under a different prefix /
#     machine never matches. We key BOTH sides by the prefix-independent
#     "<synset>/<filename>" (the last two path components), and rebuild the
#     cache if an existing LMDB does not resolve under that scheme.
#   * (1a) Fork-safety: an lmdb env opened pre-fork and shared across
#     DataLoader workers / torchrun ranks returns corrupt buffers
#     (PIL.UnidentifiedImageError). We open the env lazily per process and copy
#     bytes out inside the transaction.
#   * Concurrent first-run build race: 4 ranks x N workers all entering the
#     build branch at once corrupts the LMDB. We serialize the build behind an
#     exclusive file lock.
_IMAGENET_LOADER_SRC = '''class _ForkSafeLmdb:
    """Lazily open the LMDB env once per process (keyed by PID).

    An lmdb env opened in the parent and shared across fork()ed DataLoader
    workers / torchrun ranks hands back corrupt buffers (UnidentifiedImageError)
    because the children inherit the parent's mmap + reader-slot state. Opening
    lazily per process makes reads fork-safe. Exposes .begin() (all the loader
    needs) and is picklable (the live env is dropped on pickle so it re-opens
    in the worker process).
    """

    def __init__(self, lmdb_path):
        self._lmdb_path = lmdb_path
        self._env = None
        self._pid = None

    def _ensure(self):
        pid = os.getpid()
        if self._env is None or self._pid != pid:
            self._env = lmdb.open(
                self._lmdb_path, readonly=True, max_readers=256,
                lock=False, readahead=False, meminit=False,
            )
            self._pid = pid
        return self._env

    def begin(self, *args, **kwargs):
        return self._ensure().begin(*args, **kwargs)

    def __getstate__(self):
        return {"_lmdb_path": self._lmdb_path, "_env": None, "_pid": None}

    def __setstate__(self, state):
        self.__dict__.update(state)


def _canonical_lmdb_key(path):
    """Prefix-independent LMDB key: '<synset>/<filename>' (last two path parts).

    Upstream keys the LMDB by the absolute build-time path returned by
    datasets.ImageFolder(root).imgs. That breaks whenever the LMDB is built
    under one path prefix and queried under another (relative-vs-absolute, a
    different mount, or a pre-staged snapshot from another machine), making
    every txn.get() return None. Keying by the last two components makes build
    and query agree regardless of prefix.
    """
    p = str(path).replace("\\\\", "/").rstrip("/")
    parts = p.split("/")
    return "/".join(parts[-2:]) if len(parts) >= 2 else p


def _is_real_image(path):
    """Reject macOS AppleDouble junk (``__MACOSX/`` dirs, ``._*`` stubs,
    ``.DS_Store``). The mlx-vision/imagenet-1k val.zip was packaged on macOS and
    ships a 212-byte ``._<img>.JPEG`` stub beside every real JPEG; ImageFolder
    otherwise ingests them (they end in .JPEG) and the first-run full-val FID
    reference pass crashes on the first one (PIL.UnidentifiedImageError) ->
    best_fid_Imagenet missing -> the whole gmean score collapses to 0."""
    parts = str(path).replace("\\\\", "/").split("/")
    base = parts[-1]
    if base.startswith("._") or base == ".DS_Store":
        return False
    return "__MACOSX" not in parts


def _drop_macos_junk(data_set):
    """Strip macOS AppleDouble entries from an ImageFolder so neither the LMDB
    build nor the full-val FID reference pass ever touches a non-image file."""
    imgs = [(p, c) for (p, c) in data_set.imgs if _is_real_image(p)]
    data_set.imgs = imgs
    data_set.samples = imgs
    if hasattr(data_set, "targets"):
        data_set.targets = [c for (_p, c) in imgs]


def lmdb_loader(path, lmdb_data):
    # Copy bytes out *inside* the transaction (buffers=True returns a memoryview
    # backed by the mmap; using it after the txn closes / across a fork yields
    # corrupt data). Try the canonical key first, fall back to the raw path for
    # back-compat with old full-path-keyed LMDBs.
    with lmdb_data.begin(write=False, buffers=True) as txn:
        bytedata = txn.get(_canonical_lmdb_key(path).encode())
        if bytedata is None:
            bytedata = txn.get(str(path).encode())
        if bytedata is None:
            raise KeyError(
                "LMDB has no entry for %r (canonical key %r). The cached "
                "*_faster_imagefolder.lmdb(.pt) was built under a different "
                "path layout; delete it so it is rebuilt."
                % (path, _canonical_lmdb_key(path))
            )
        bytedata = bytes(bytedata)
    img = Image.open(io.BytesIO(bytedata))
    return img.convert("RGB")


def _lmdb_resolves(lmdb_path, probe_path):
    """True iff the existing LMDB resolves probe_path under the canonical key."""
    env = lmdb.open(lmdb_path, readonly=True, max_readers=1, lock=False,
                    readahead=False, meminit=False)
    try:
        with env.begin(write=False) as txn:
            return txn.get(_canonical_lmdb_key(probe_path).encode()) is not None
    finally:
        env.close()


def _build_lmdb_dataset(root, transform=None, target_transform=None, loader=lmdb_loader):
    """
    You can create this dataloader using:
    train_data = _build_lmdb_dataset(traindir, transform=train_transform)
    valid_data = _build_lmdb_dataset(validdir, transform=val_transform)
    """
    import fcntl

    root = str(root)
    if root.endswith("/"):
        root = root[:-1]
    pt_path = os.path.join(root + "_faster_imagefolder.lmdb.pt")
    lmdb_path = os.path.join(root + "_faster_imagefolder.lmdb")

    # Serialize the (slow, single-writer) build across torchrun ranks and
    # DataLoader workers via an exclusive file lock: only the lock holder
    # builds; the rest block, then load the finished artifact. Prevents the
    # concurrent-build races that corrupt the LMDB under num_workers>0 / DDP.
    lock_path = root + "_faster_imagefolder.lmdb.build.lock"
    lock_dir = os.path.dirname(os.path.abspath(lock_path))
    if lock_dir:
        os.makedirs(lock_dir, exist_ok=True)
    lock_file = open(lock_path, "w")
    fcntl.flock(lock_file, fcntl.LOCK_EX)
    try:
        have_cache = os.path.isfile(pt_path) and os.path.isdir(lmdb_path)
        if have_cache:
            data_set = torch.load(pt_path, weights_only=False)
            _drop_macos_junk(data_set)
            # Stale-cache guard: a snapshot built under a foreign path prefix
            # has keys that never match our queries. Probe the first image; if
            # it does not resolve, drop the cache and rebuild from disk.
            if data_set.imgs and not _lmdb_resolves(lmdb_path, data_set.imgs[0][0]):
                import shutil
                shutil.rmtree(lmdb_path, ignore_errors=True)
                try:
                    os.remove(pt_path)
                except OSError:
                    pass
                have_cache = False
        if not have_cache:
            data_set = datasets.ImageFolder(root, None, None, None)
            _drop_macos_junk(data_set)
            torch.save(data_set, pt_path, pickle_protocol=4)
            env = lmdb.open(lmdb_path, map_size=int(1e12))
            with env.begin(write=True) as txn:
                for _path, class_index in data_set.imgs:
                    with open(_path, "rb") as f:
                        data = f.read()
                    txn.put(_canonical_lmdb_key(_path).encode("ascii"), data)
            env.sync()
            env.close()
    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()

    # Per-process lazy env (fork-safe). data_set.lmdb_data.begin(...) is all the
    # loader uses.
    data_set.lmdb_data = _ForkSafeLmdb(lmdb_path)
    # reset transform and target_transform
    data_set.samples = data_set.imgs
    data_set.transform = transform
    data_set.target_transform = target_transform
    data_set.loader = lambda path: loader(path, data_set.lmdb_data)

    return data_set
'''

OPS = [
    # 1. Make Inception model path absolute relative to the repo's assets dir.
    {
        "op": "replace",
        "file": "dbim-codebase/evaluations/feature_extractor.py",
        "start_line": 19,
        "end_line": 19,
        "content": '        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")\n',
    },
    # 2. Respect the CLI --num_samples instead of overwriting it with the full
    #    dataset length. Without this, --num_samples is ignored and we always
    #    sample the entire train set (138k for e2h).
    {
        "op": "replace",
        "file": "dbim-codebase/sample.py",
        "start_line": 80,
        "end_line": 80,
        "content": "    args.num_samples = min(args.num_samples, len(dataloader.dataset))\n",
    },
    # 3. Add an early-exit to the sampling loop so runs don't generate images
    #    past args.num_samples. Replaces the trailing rank-0 log line with a
    #    log + break-on-enough block.
    {
        "op": "replace",
        "file": "dbim-codebase/sample.py",
        "start_line": 154,
        "end_line": 155,
        "content": (
            "        if dist.get_rank() == 0:\n"
            "            logger.log(f\"sampled {num} images\")\n"
            "        if num >= args.num_samples:\n"
            "            break\n"
        ),
    },
    # 4b. Only compute the scored metric (FID). Upstream evaluate.sh also runs
    #     LPIPS/SSIM (e2h/diode) and Inception Score (imagenet), none of which
    #     are in score_spec.py or parsed by parser.py. They are pure overhead and
    #     LPIPS/SSIM `assert ref_batch.shape == sample_batch.shape` crashes when
    #     num_samples != the reference-set size (which it is here), making the
    #     command exit non-zero. Drop the `--metric lpips` and `--metric is`
    #     calls. Replaces lines 47-54 (the trailing metric if/elif block).
    #     NOTE: listed BEFORE the line-44 op below so same-file pre_edit ops
    #     apply bottom-to-top (pristine line numbers).
    {
        "op": "replace",
        "file": "dbim-codebase/scripts/evaluate.sh",
        "start_line": 47,
        "end_line": 54,
        "content": (
            'if [[ $DATASET_NAME == "e2h" || $DATASET_NAME == "diode" ]]; then\n'
            "    python evaluations/evaluator.py $REF_PATH $SAMPLE_PATH --metric fid\n"
            'elif [[ $DATASET_NAME == "imagenet_inpaint_center" ]]; then\n'
            "    LABEL_PATH=${SAMPLE_DIR}/${LABEL_NAME}\n"
            "    python evaluation/compute_metrices_imagenet.py --ckpt $SAMPLE_PATH --label $LABEL_PATH --dataset-dir $DATA_DIR\n"
            "fi\n"
        ),
    },
    # 4. Make the evaluator find whatever sample file was actually written,
    #    instead of looking for a hardcoded size (138567/16502/10000) and a
    #    specific nfe number. Heun/high-order solvers return a different `nfe`
    #    than the --steps arg, and num_samples is now configurable, so both
    #    the size and nfe can differ from the hardcoded SAMPLE_NAME.
    {
        "op": "replace",
        "file": "dbim-codebase/scripts/evaluate.sh",
        "start_line": 44,
        "end_line": 45,
        "content": (
            "SAMPLE_DIR=workdir/${PREFIX}/split=${SPLIT}/${SAMPLER}/steps=${N}\n"
            "SAMPLE_PATH=$(ls ${SAMPLE_DIR}/samples_*.npz 2>/dev/null | head -1)\n"
            'if [ -z "$SAMPLE_PATH" ]; then\n'
            '    echo "ERROR: no samples_*.npz found under $SAMPLE_DIR" >&2\n'
            "    exit 1\n"
            "fi\n"
            "# Imagenet evaluator also needs a labels_*.npz — glob it too (size/nfe vary).\n"
            "LABEL_NAME=$(ls ${SAMPLE_DIR}/labels_*.npz 2>/dev/null | head -1 | xargs -n1 basename)\n"
        ),
    },
    # 5. Forward a --num_samples CLI arg to sample.py (read from the `num_samples`
    #    env var that the per-task run scripts export). No-op if the var isn't set.
    {
        "op": "replace",
        "file": "dbim-codebase/scripts/sample.sh",
        "start_line": 53,
        "end_line": 59,
        "content": (
            "torchrun $run_args sample.py --steps $N --sampler $GEN_SAMPLER --batch_size $BS \\\n"
            " --model_path $MODEL_PATH --class_cond $CLASS_COND --noise_schedule $PRED \\\n"
            ' ${BETA_D:+ --beta_d="${BETA_D}"} ${BETA_MIN:+ --beta_min="${BETA_MIN}"} ${BETA_MAX:+ --beta_max="${BETA_MAX}"} \\\n'
            " --condition_mode=$COND  --sigma_max=$SIGMA_MAX --sigma_min=$SIGMA_MIN \\\n"
            " --dropout $DROPOUT --image_size $IMG_SIZE --num_channels $NUM_CH  --num_res_blocks $NUM_RES_BLOCKS \\\n"
            " --use_new_attention_order $ATTN_TYPE --data_dir=$DATA_DIR --dataset=$DATASET --split $SPLIT \\\n"
            ' ${num_samples:+ --num_samples="${num_samples}"} ${SEED:+ --seed="${SEED}"} \\\n'
            ' ${CHURN_STEP_RATIO:+ --churn_step_ratio="${CHURN_STEP_RATIO}"} \\\n'
        ),
    },
    # 6b. Replace the ImageNet LMDB loader + builder with a prefix-independent,
    #     fork-safe, race-free version. See _IMAGENET_LOADER_SRC above for the
    #     full rationale (fixes the all-keys-miss crash that zeroed the Imagenet
    #     FID, plus DataLoader fork corruption and concurrent-build races).
    #     Replaces lines 24-67 (functions `lmdb_loader` and `_build_lmdb_dataset`).
    {
        "op": "replace",
        "file": "dbim-codebase/datasets/imagenet_inpaint.py",
        "start_line": 24,
        "end_line": 67,
        "content": _IMAGENET_LOADER_SRC,
    },
    # 6d. Enforce NFE budget on the denoiser inside karras_sample.
    # Agents were cheating (double-denoise / Heun corrector -> 6-9 actual calls
    # while reporting nfe=len(ts)-1). Wrap the closure so it:
    #   * counts every actual diffusion.denoise invocation
    #   * raises RuntimeError on call N+1 where N = steps+1 (true NFE budget)
    #   * prints "ACTUAL_NFE: <n> / EXPECTED_NFE: <n>" for the parser
    # All agent edits go through this closure — there's no other way to call
    # the model. A raised error stops sample.py mid-run, no samples file is
    # written, evaluator finds nothing, agent gets feedback "budget exceeded".
    {
        "op": "replace",
        "file": "dbim-codebase/ddbm/karras_diffusion.py",
        "start_line": 275,
        "end_line": 279,
        "content": (
            "    _nfe_budget = steps + 1\n"
            "    _nfe_counter = [0]\n"
            "    def denoiser(x_t, sigma):\n"
            "        if _nfe_counter[0] >= _nfe_budget:\n"
            "            raise RuntimeError(\n"
            "                f'NFE_BUDGET_EXCEEDED: used {_nfe_counter[0]+1} denoiser calls '\n"
            "                f'but budget is {_nfe_budget}. Do not double-denoise / Heun-correct '\n"
            "                f'beyond the allowed NFE.'\n"
            "            )\n"
            "        _nfe_counter[0] += 1\n"
            "        _, denoised, _ = diffusion.denoise(model, x_t, sigma, **model_kwargs)\n"
            "        if clip_denoised:\n"
            "            denoised = denoised.clamp(-1, 1)\n"
            "        return denoised\n"
        ),
    },
    # (Removed OP 6e: it used original line numbers that were stale after
    # OP 6d inserted 9 lines, overwriting `return denoised` and breaking
    # the denoiser closure. RuntimeError from OP 6d already contains the
    # string "NFE_BUDGET_EXCEEDED" which the parser detects from stderr.)
    # 6c. evaluation/resnet.py imports `from ipdb import set_trace as debug` only
    #     for debugging. ipdb transitively needs the `decorator` package which
    #     is not in our install_cmds. Strip the ipdb import; `debug` isn't used
    #     anywhere in the runtime path.
    {
        "op": "replace",
        "file": "dbim-codebase/evaluation/resnet.py",
        "start_line": 14,
        "end_line": 14,
        "content": "\n",
    },
    # 6. For imagenet_inpaint sampling: reuse the val10k testset as trainset/valset
    #    placeholders instead of opening the val lmdb three times. The train lmdb is
    #    a symlink to val (we only downloaded val), so two opens of the same file
    #    raises `lmdb.Error: environment already open in this process`. sample.py
    #    only iterates `all_dataloaders[2]` (the testset) when split=test, so the
    #    other two loaders never yield — they just need to be constructible.
    {
        "op": "replace",
        "file": "dbim-codebase/datasets/__init__.py",
        "start_line": 217,
        "end_line": 226,
        "content": (
            '    elif "imagenet_inpaint" in dataset:\n'
            '        corrupt_type = dataset.split("_")[-1]\n'
            '        assert corrupt_type in ["center", "freeform2030"]\n'
            "        from .imagenet_inpaint import ImageNetInpaintingDataset, InpaintingVal10kSubset\n"
            "\n"
            "        if include_test:\n"
            "            # Sampling path: only the testset (val10k subset) is read.\n"
            "            # Reuse it for train/val slots to avoid re-opening the lmdb.\n"
            "            testset = InpaintingVal10kSubset(root, image_size, corrupt_type)\n"
            "            trainset = testset\n"
            "            valset = testset\n"
            "        else:\n"
            "            trainset = ImageNetInpaintingDataset(root, image_size, corrupt_type, train=True)\n"
            "            valset = ImageNetInpaintingDataset(root, image_size, corrupt_type, train=False)\n"
        ),
    },
    # 7. Match torchrun's nproc_per_node to the actual number of GPUs
    #    visible inside the container. Upstream sample.sh hardcodes
    #    `CUDA_VISIBLE_DEVICES=0,1,...,7` and `--nproc_per_node 8`, so on a
    #    smaller allocation torchrun spins up 8 ranks but only N<8 GPUs
    #    are visible — ranks past N-1 abort with `CUDA error: invalid
    #    device ordinal`. Detect via nvidia-smi (always present when
    #    docker --gpus / apptainer --nv is used). Falls back to 8 if the
    #    probe fails so single-host runs without nvidia-smi still work.
    {
        "op": "replace",
        "file": "dbim-codebase/scripts/sample.sh",
        "start_line": 11,
        "end_line": 13,
        "content": (
            "if command -v nvidia-smi >/dev/null 2>&1; then\n"
            "    NGPU=$(nvidia-smi --list-gpus 2>/dev/null | grep -c '^GPU ')\n"
            "fi\n"
            "if [ -z \"${NGPU:-}\" ] || [ \"$NGPU\" -lt 1 ]; then\n"
            "    NGPU=8\n"
            "fi\n"
            "run_args=\"--nproc_per_node $NGPU \\\n"
            "          --master_port 29511\"\n"
        ),
    },
]
