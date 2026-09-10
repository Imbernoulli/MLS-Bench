#!/bin/bash
# Verifier-only runtime patch for the meta-inner-loop-optimizer scaffold
# (learn2learn MetaDataset bookkeeping-cache write race).
#
# The task template tasks/meta-inner-loop-optimizer/edits/custom_template.py
# is fixed at the source, but the rendered scaffold is baked into the
# per-task image as /workspace/learn2learn/custom_maml.py. learn2learn's
# MetaDataset.load_bookkeeping() is a TOCTOU exists-check followed by a
# truncating pickle write; natively each (label, seed) run gets its own
# container FS, but under Harbor all (label, seed) eval subprocesses share
# ONE container FS, so concurrent writers/readers hit UnpicklingError or
# leave a persistently corrupt cache. This patch retrofits the template's
# flock + self-heal wrapper onto a stale image-baked scaffold at eval time.
#
# Detection AND application are EOF-anchored against the file's protected
# tail. tests/meta/config.json declares exactly one editable range for
# custom_maml.py (lines 177-254, the InnerLoopOptimizer class), so every
# byte below it is protected and always terminates the file. The patch
# compares the file's tail against the two byte-exact known states of that
# protected region — the pre-fix render's tail and the fixed template's
# tail — and swaps stale tail -> fixed tail in a single write. Content the
# agent puts inside the editable range can never influence the decision
# (a whole-file marker search could previously be forged from the editable
# region, suppressing the helper insert while the call-site rewrite still
# applied -> NameError at eval); a tail matching neither known state
# (unexpected variant / tampered workspace) is left untouched,
# all-or-nothing. Runs after the edit-range guard (score_task.py guard
# precedes run-evals), under an flock so concurrent (label, seed) eval
# processes don't race the rewrite itself.

_l2l_patch_lock="${L2L_RUNTIME_PATCH_LOCK:-.l2l_runtime_patch.lock}"
{
if command -v flock >/dev/null 2>&1; then
    flock 9
fi

python3 - <<'PY' || echo "[_runtime_patch] WARNING: custom_maml.py patch failed" >&2
from pathlib import Path

path = Path("learn2learn/custom_maml.py")
if path.exists():
    text = path.read_text()

    helper = '''# =====================================================================
# FIXED: Concurrency-safe taskset construction
# =====================================================================
# NOTE: defined below the editable class on purpose — config.json's editable
# range for this file (the InnerLoopOptimizer class) is line-anchored, so
# fixed scaffolding added above it would shift the range.
def _bookkeeping_paths(dataset_name: str, root: str) -> List[str]:
    """learn2learn's bookkeeping cache pickles for this dataset's splits."""
    if dataset_name == "mini_imagenet":
        return [os.path.join(root, "mini-imagenet-bookkeeping-%s.pkl" % m)
                for m in ("train", "validation", "test")]
    if dataset_name == "cifar_fs":
        # CIFARFS maps mode "validation" -> "val" in its bookkeeping filename.
        return [os.path.join(root, "cifarfs-%s-bookkeeping.pkl" % m)
                for m in ("train", "val", "test")]
    return []


def get_tasksets_locked(dataset_name: str, n_way: int, n_shot: int, n_query: int,
                        root: str = os.environ.get("L2L_DATA_ROOT",
                                                   "/workspace/l2l_data")):
    """Serialize taskset construction across concurrent evaluation runs.

    learn2learn's ``MetaDataset.load_bookkeeping`` is an exists-check followed
    by a truncating pickle write of the bookkeeping cache — a TOCTOU race.
    When several (setting, seed) evaluation processes share one filesystem,
    a reader can observe a partially-written pickle (``UnpicklingError`` /
    ``EOFError``) or the racing writers can leave a persistently corrupt
    cache behind.

    This wrapper holds an inter-process ``flock`` on a lockfile next to the
    data root for the whole construct-or-load of all three splits, so exactly
    one process builds each cache and the rest load the finished pickle. A
    corrupt cache (e.g. left by a previously crashed writer) is self-healed
    by deleting this dataset's cache pickles and retrying once, still under
    the lock.
    """
    import fcntl

    try:
        os.makedirs(root, exist_ok=True)
        lock_file = open(os.path.join(root, ".l2l_bookkeeping.lock"), "a+")
    except OSError:
        # Read-only data root: the caches must already exist, constructions
        # only read them, and no serialization is needed.
        lock_file = None
    try:
        if lock_file is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            return get_tasksets(dataset_name, n_way, n_shot, n_query, root=root)
        except Exception:
            # Corrupt bookkeeping caches surface as unpickling errors inside
            # MetaDataset.__init__. Delete them and rebuild once while still
            # holding the lock; unrelated errors re-raise from the retry (and
            # immediately below when there was no cache file to remove).
            removed = False
            for pkl_path in _bookkeeping_paths(dataset_name, root):
                try:
                    os.remove(pkl_path)
                    removed = True
                except OSError:
                    pass
            if not removed:
                raise
            return get_tasksets(dataset_name, n_way, n_shot, n_query, root=root)
    finally:
        if lock_file is not None:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            lock_file.close()


'''
    anchor = (
        "# =====================================================================\n"
        "# FIXED: Meta-Training and Evaluation Loop\n"
        "# =====================================================================\n"
    )
    OLD_CALL = (
        "    # Load tasksets\n"
        "    train_tasks, val_tasks, test_tasks = get_tasksets(\n"
        "        DATASET_NAME, N_WAY, N_SHOT, N_QUERY\n"
        "    )\n"
    )
    NEW_CALL = (
        "    # Load tasksets (serialized against concurrent runs sharing this FS)\n"
        "    train_tasks, val_tasks, test_tasks = get_tasksets_locked(\n"
        "        DATASET_NAME, N_WAY, N_SHOT, N_QUERY\n"
        "    )\n"
    )

    # tests/meta/config.json declares ONE editable range for this file:
    # lines 177-254, the InnerLoopOptimizer class (hardcoded here — keep in
    # sync with config.json). Both rewrite sites sit BELOW that range, in the
    # protected region that runs to EOF, so on any workspace whose protected
    # bytes are intact the file's tail IS that protected region. STALE_TAIL
    # is the byte-exact protected tail (pristine lines 255..EOF) of the
    # pre-fix render; FIXED_TAIL — derived from it by the two anchored
    # replacements — is the same tail in the fixed template. Comparing the
    # file's tail against these two constants means nothing the agent writes
    # inside the editable range (which always lies ABOVE the tail) can
    # suppress, trigger, or misplace the patch.
    STALE_TAIL = '''

# =====================================================================
# FIXED: Meta-Training and Evaluation Loop
# =====================================================================
def meta_train_step(model, inner_opt, meta_optimizer,
                    taskset, n_way, n_shot, n_query, meta_batch_size,
                    inner_steps, device):
    """One meta-training iteration: sample tasks, adapt, compute meta-loss."""
    meta_train_loss = 0.0
    meta_train_acc = 0.0

    for _ in range(meta_batch_size):
        # Clone model for this task
        learner = l2l.clone_module(model)

        # Sample a task
        task_data = taskset.sample()
        data, labels = task_data
        data, labels = data.to(device), labels.to(device)

        # Split into support / query
        support_x, support_y, query_x, query_y = split_support_query(
            data, labels, n_way, n_shot
        )

        # Inner-loop adaptation (uses the shared inner_opt instance)
        learner = inner_opt.adapt(learner, support_x, support_y, inner_steps)

        # Evaluate on query set (for meta-gradient)
        loss, acc = compute_loss_and_acc(learner, query_x, query_y)
        meta_train_loss += loss
        meta_train_acc += acc

    meta_train_loss /= meta_batch_size
    meta_train_acc /= meta_batch_size

    # Meta-update
    meta_optimizer.zero_grad()
    meta_train_loss.backward()
    meta_optimizer.step()

    return meta_train_loss.item(), meta_train_acc


def meta_evaluate(model, inner_opt, taskset,
                  n_way, n_shot, n_query, n_tasks, inner_steps, device):
    """Evaluate on a set of tasks."""
    accs = []
    for _ in range(n_tasks):
        learner = l2l.clone_module(model)

        task_data = taskset.sample()
        data, labels = task_data
        data, labels = data.to(device), labels.to(device)

        support_x, support_y, query_x, query_y = split_support_query(
            data, labels, n_way, n_shot
        )

        learner = inner_opt.adapt(learner, support_x, support_y, inner_steps)

        with torch.no_grad():
            _, acc = compute_loss_and_acc(learner, query_x, query_y)
        accs.append(acc)

    mean_acc = np.mean(accs)
    ci95 = 1.96 * np.std(accs) / np.sqrt(len(accs))
    return mean_acc, ci95


# =====================================================================
# FIXED: Main Script
# =====================================================================
if __name__ == "__main__":
    # Reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Dataset: {DATASET_NAME}, N-way: {N_WAY}, N-shot: {N_SHOT}, Seed: {SEED}", flush=True)
    print(f"Setting: {SETTING}", flush=True)
    print(f"Meta-LR: {META_LR}, Inner-LR: {INNER_LR}", flush=True)
    print(f"Inner steps train/test: {INNER_STEPS_TRAIN}/{INNER_STEPS_TEST}", flush=True)

    # Load tasksets
    train_tasks, val_tasks, test_tasks = get_tasksets(
        DATASET_NAME, N_WAY, N_SHOT, N_QUERY
    )

    # Build model
    model = make_model(N_WAY).to(DEVICE)

    # ── FIXED: Parameter count check ────────────────────────────────
    # Budget: CNN4 model (~112K) + inner-loop optimizer learnable params
    # Meta-SGD adds one scalar per parameter (~112K extra).
    # Budget is 1.2x of (model params + Meta-SGD optimizer params).
    _model_params = sum(p.numel() for p in model.parameters())
    _optimizer_budget = _model_params  # Meta-SGD needs one LR per param
    _budget = int((_model_params + _optimizer_budget) * 1.2)

    # Create inner-loop optimizer (persistent across all iterations)
    inner_opt = InnerLoopOptimizer(model, INNER_LR)
    _opt_params = sum(p.numel() for p in inner_opt.meta_parameters())
    _total_params = _model_params + _opt_params
    print(f"Model params: {_model_params:,}, Optimizer params: {_opt_params:,}, "
          f"Total: {_total_params:,} (budget: {_budget:,})", flush=True)
    # ────────────────────────────────────────────────────────────────

    # Collect all meta-learnable parameters: model params + optimizer params
    all_meta_params = list(model.parameters()) + list(inner_opt.meta_parameters())
    meta_optimizer = torch.optim.Adam(all_meta_params, lr=META_LR)

    # Meta-training loop
    best_val_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())
    best_inner_meta_state = [
        p.detach().clone() for p in inner_opt.meta_parameters()
    ]

    for iteration in range(1, N_META_ITERS + 1):
        model.train()
        train_loss, train_acc = meta_train_step(
            model, inner_opt, meta_optimizer,
            train_tasks, N_WAY, N_SHOT, N_QUERY, META_BATCH_SIZE,
            INNER_STEPS_TRAIN, DEVICE,
        )

        if iteration % EVAL_INTERVAL == 0:
            model.eval()
            val_acc, val_ci = meta_evaluate(
                model, inner_opt, val_tasks,
                N_WAY, N_SHOT, N_QUERY, N_EVAL_TASKS,
                INNER_STEPS_TEST, DEVICE,
            )
            print(
                f"TRAIN_METRICS iter={iteration} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_acc={val_acc:.4f} val_ci95={val_ci:.4f}",
                flush=True,
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                best_inner_meta_state = [
                    p.detach().clone() for p in inner_opt.meta_parameters()
                ]
                print(f"  New best val accuracy: {val_acc:.4f} +/- {val_ci:.4f}", flush=True)

    # Load best model and evaluate on test set
    model.load_state_dict(best_state)
    with torch.no_grad():
        for p, saved in zip(inner_opt.meta_parameters(), best_inner_meta_state):
            p.copy_(saved.to(device=p.device, dtype=p.dtype))
    model.eval()
    TEST_RNG_SEED = 0xBEEF
    random.seed(TEST_RNG_SEED)
    np.random.seed(TEST_RNG_SEED)
    torch.manual_seed(TEST_RNG_SEED)
    torch.cuda.manual_seed_all(TEST_RNG_SEED)
    test_acc, test_ci = meta_evaluate(
        model, inner_opt, test_tasks,
        N_WAY, N_SHOT, N_QUERY, N_TEST_TASKS,
        INNER_STEPS_TEST, DEVICE,
    )
    print(f"TEST_METRICS accuracy={test_acc:.4f} ci95={test_ci:.4f}", flush=True)
    print(f"Test accuracy: {100 * test_acc:.2f}% +/- {100 * test_ci:.2f}%", flush=True)
'''

    if STALE_TAIL.count(anchor) == 1 and STALE_TAIL.count(OLD_CALL) == 1:
        FIXED_TAIL = STALE_TAIL.replace(anchor, helper + anchor, 1)
        FIXED_TAIL = FIXED_TAIL.replace(OLD_CALL, NEW_CALL, 1)
        if text.endswith(FIXED_TAIL):
            pass  # fix already present in the protected tail — clean no-op
        elif text.endswith(STALE_TAIL):
            # Atomic helper-insert + call-site rewrite: swap the whole
            # protected tail in one write. Bytes above the tail (including
            # the agent's editable region) are byte-for-byte untouched.
            path.write_text(text[:len(text) - len(STALE_TAIL)] + FIXED_TAIL)
        # else: the protected tail matches neither known state (unexpected
        # template variant / tampered workspace) — leave the file untouched
        # rather than risk a partial or misplaced rewrite.
PY
} 9>"${_l2l_patch_lock}"
