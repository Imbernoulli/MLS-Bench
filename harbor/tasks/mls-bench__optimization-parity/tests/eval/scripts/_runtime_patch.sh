#!/bin/bash
# Verifier-only runtime patch for optimization-parity.
#
# The agent workspace image bakes pytorch-examples/optimization_parity/
# custom_strategy.py from environment/_scaffold at image-build time. A stale
# image may still carry the pre-fix FIXED driver code, whose
# _load_all_train_labels deleted the staged label blobs unconditionally after
# the first load — natively (blobs staged once per workspace) that crashed
# every test after the first. Splice in the fixed blocks (scrub gated on
# MLSBENCH_EPHEMERAL_INPUTS=1) before the eval runs. Every replacement is an
# anchored exact-string match against pre-fix FIXED-region text only (the
# agent's editable region, lines 306-341, is never touched): it applies only
# when the pre-fix block is present and is a no-op on an up-to-date workspace.

_parity_patch_lock="${PARITY_RUNTIME_PATCH_LOCK:-/workspace/.parity_runtime_patch.lock}"
{
if command -v flock >/dev/null 2>&1; then
    flock 9
fi

python3 - <<'PY'
import os
import tempfile

TARGET = "/workspace/pytorch-examples/optimization_parity/custom_strategy.py"

REPLACEMENTS = [
    (
        '''NOTE: The hidden parity secret S, the training-pool LABELS, and the held-out
test labels are NOT visible to your editable hooks. The harness pre-generates
the (unlabeled) training inputs and their labels; a FIXED driver loads the
labels into its own memory, deletes them from disk, then hands your
``make_dataset`` only the UNLABELED pool and attaches the held-out labels to the
rows you pick. Your hooks therefore only ever see binary inputs — never a label,
never the secret subset S, never the test labels. The runner trains your model
and emits its predictions on a held-out test set; the host regenerates the test
labels and computes test accuracy. A strategy must make gradient training learn
the parity — it cannot recover the secret from labels.
''',
        '''NOTE: The hidden parity secret S, the training-pool LABELS, and the held-out
test labels are NOT visible to your editable hooks. The harness pre-generates
the (unlabeled) training inputs and their labels; a FIXED driver loads the
labels into its own memory (scrubbing them from disk when the harness marks
them ephemeral), then hands your ``make_dataset`` only the UNLABELED pool and
attaches the held-out labels to the rows you pick. Your hooks only ever see
binary inputs — never a label, never S, never the test labels. The runner
trains your model and emits its predictions on a held-out test set; the host
regenerates the test labels and computes accuracy. A strategy must make
gradient training learn the parity — it cannot recover the secret from labels.
''',
    ),
    (
        '''# FIXED: held-out input loading (the harness pre-generates these; the
# secret and the test labels are never present in this process). The
# training-pool LABELS are loaded here, in fixed code, and then scrubbed
# from disk BEFORE any editable hook runs — so make_dataset() below only
# ever sees the UNLABELED pool.
''',
        '''# FIXED: held-out input loading (the harness pre-generates these; the
# secret and the test labels are never present in this process). The
# training-pool LABELS are loaded here, in fixed code — and scrubbed from
# disk when the harness marks them ephemeral (MLSBENCH_EPHEMERAL_INPUTS=1)
# — so make_dataset() below only ever sees the UNLABELED pool.
''',
    ),
    (
        '''    This is FIXED code, called only by ``_load_all_train_labels`` below, which
    immediately deletes the on-disk blob afterward. It is never invoked from an
    editable hook.
    """
''',
        '''    This is FIXED code, called only by ``_load_all_train_labels`` below, which
    deletes the on-disk blob afterward when the harness marks the inputs as
    ephemeral. It is never invoked from an editable hook.
    """
''',
    ),
    # Gate the label-blob scrub on MLSBENCH_EPHEMERAL_INPUTS=1 (exported by
    # these eval scripts; apply.py re-materializes the blobs every evaluation).
    (
        '''def _load_all_train_labels(config: TaskConfig, seed: int) -> dict[int, torch.Tensor]:
    """Load every hidden secret's training-pool labels into memory, then DELETE
    the on-disk label blobs.

    After this returns, the labels exist only inside this fixed driver's local
    scope; the ``.labels.b64`` files are gone from the workspace, so the editable
    ``make_dataset`` hook (which runs later) cannot open them to recover the
    hidden secret. This is what keeps parity honest: the strategy must help
    gradient training learn the parity rather than solve the secret from labels.
    """
    labels: dict[int, torch.Tensor] = {}
    for secret_index in range(config.num_hidden_secrets):
        labels[secret_index] = load_train_labels(config, seed, secret_index)
    tag = _config_tag(config)
    inputs_dir = _inputs_dir()
    for secret_index in range(config.num_hidden_secrets):
        blob = os.path.join(inputs_dir, f"{tag}_seed{seed}_s{secret_index}.labels.b64")
        try:
            os.remove(blob)
        except OSError:
            pass
    return labels
''',
        '''def _load_all_train_labels(config: TaskConfig, seed: int) -> dict[int, torch.Tensor]:
    """Load every hidden secret's training-pool labels into memory, then DELETE
    the on-disk label blobs when the harness marks the materialized inputs as
    ephemeral (MLSBENCH_EPHEMERAL_INPUTS=1, i.e. re-created before every
    evaluation). The delete keeps parity honest there: the editable
    ``make_dataset`` hook (which runs later) cannot reopen the blobs to recover
    the hidden secret, so the strategy must help gradient training learn the
    parity. Natively (no ENV set) the blobs persist — they are staged once per
    workspace and must survive across evaluations."""
    labels: dict[int, torch.Tensor] = {}
    for secret_index in range(config.num_hidden_secrets):
        labels[secret_index] = load_train_labels(config, seed, secret_index)
    if os.environ.get("MLSBENCH_EPHEMERAL_INPUTS") == "1":
        import glob as _glob
        tag = _config_tag(config)
        pattern = os.path.join(_inputs_dir(), f"{tag}_seed{seed}_s*.labels.b64")
        for blob in _glob.glob(pattern):
            try:
                os.remove(blob)
            except OSError:
                pass
    return labels
''',
    ),
    (
        '''    # FIXED: load every secret's pool labels into memory and delete the on-disk
    # blobs before any editable hook runs. Labels now live only in this local.
''',
        '''    # FIXED: load every secret's pool labels into memory (scrubbing the blobs
    # when the harness marks them ephemeral) before any editable hook runs.
''',
    ),
    # ── round-7: fixed-entry wrapper protocol (preload + unlink BEFORE the
    # editable module is imported; loaders consult _PRELOADED_INPUTS). The
    # pairs above bring a pre-fix image to the round-2 text; these bring the
    # round-2 text to the current shipped template, byte-identically.
    (
        '''def load_train_labels(config: TaskConfig, seed: int, secret_index: int) -> torch.Tensor:
    """Load the bit-packed training-pool labels for one hidden secret.

    Only the labels are provided (the secret that produced them is held out).
    The labels are bit-packed for one row per training example over the full
    ``max_train_examples`` pool; unpack to a float tensor in {0, 1}.

    This is FIXED code, called only by ``_load_all_train_labels`` below, which
    deletes the on-disk blob afterward when the harness marks the inputs as
    ephemeral. It is never invoked from an editable hook.
    """
    import numpy as np

    tag = _config_tag(config)
    path = os.path.join(_inputs_dir(), f"{tag}_seed{seed}_s{secret_index}.labels.b64")
    with open(path, "r") as f:
        packed = np.frombuffer(base64.b64decode(f.read()), dtype=np.uint8)
    bits = np.unpackbits(packed)[: config.max_train_examples]
    return torch.from_numpy(bits.astype("float32"))
''',
        '''def load_train_labels(config: TaskConfig, seed: int, secret_index: int) -> torch.Tensor:
    """Load one hidden secret's bit-packed training-pool labels (only the
    labels — the secret that produced them is held out).

    Prefers the payload preloaded by the FIXED wrapper (scripts/fixed_entry.py
    reads and unlinks the blobs BEFORE this module is imported); falls back to
    the on-disk blob when launched directly. FIXED code, called only by
    ``_load_all_train_labels`` below — never from an editable hook.
    """
    import numpy as np

    name = f"{_config_tag(config)}_seed{seed}_s{secret_index}.labels.b64"
    payload = (_PRELOADED_INPUTS or {}).pop(name, None)
    if payload is None:
        with open(os.path.join(_inputs_dir(), name), "r") as f:
            payload = f.read()
    packed = np.frombuffer(base64.b64decode(payload), dtype=np.uint8)
    bits = np.unpackbits(packed)[: config.max_train_examples]
    return torch.from_numpy(bits.astype("float32"))
''',
    ),
    (
        '''# =====================================================================
# FIXED: training and prediction driver
# =====================================================================
def train_one_run(
''',
        '''# =====================================================================
# FIXED: training and prediction driver
# =====================================================================
# Set by the FIXED wrapper (scripts/fixed_entry.py) AFTER it read and
# unlinked the staged label blobs and BEFORE this module was imported:
# maps blob basename -> file content. None when the module is launched
# directly — the loaders above then read the on-disk blobs (legacy flow).
_PRELOADED_INPUTS: dict[str, str] | None = None


def train_one_run(
''',
    ),
    (
        '''    labels_by_secret = _load_all_train_labels(config, seed)

    results: list[RunResult] = []
''',
        '''    labels_by_secret = _load_all_train_labels(config, seed)
    # Drop any remaining preloaded payloads: from here on the labels live
    # only in fixed-driver locals, exactly as in the direct-launch flow.
    global _PRELOADED_INPUTS
    _PRELOADED_INPUTS = None

    results: list[RunResult] = []
''',
    ),
]

if os.path.exists(TARGET):
    with open(TARGET, "r", encoding="utf-8") as fh:
        text = fh.read()
    patched = text
    for old, new in REPLACEMENTS:
        if old in patched:
            patched = patched.replace(old, new, 1)
    if patched != text:
        target_dir = os.path.dirname(TARGET)
        fd, tmp = tempfile.mkstemp(dir=target_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(patched)
            os.replace(tmp, TARGET)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        print(f"[runtime-patch] updated {TARGET}", flush=True)
PY
} 9>"${_parity_patch_lock}"
