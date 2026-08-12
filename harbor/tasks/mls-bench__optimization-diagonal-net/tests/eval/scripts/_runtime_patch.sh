#!/bin/bash
# Verifier-only runtime patch for optimization-diagonal-net.
#
# The agent workspace image bakes RAIN/opt_diagonal_net/fixed_benchmark.py from
# environment/_scaffold at image-build time. A stale image may still carry the
# pre-fix runner, which (a) built input-blob keys WITHOUT --sigma, so the two
# concurrent d500_k10 settings raced on the SAME files, and (b) deleted the
# blobs unconditionally after loading, which natively (blobs staged once)
# crashed every test after the first. Splice in the fixed blocks before the
# eval runs. Every replacement is an anchored exact-string match: it applies
# only when the pre-fix block is present, and the whole patch is a no-op on an
# up-to-date workspace.

_diagnet_patch_lock="${DIAGNET_RUNTIME_PATCH_LOCK:-/workspace/.diagnet_runtime_patch.lock}"
{
if command -v flock >/dev/null 2>&1; then
    flock 9
fi

python3 - <<'PY'
import os
import tempfile

TARGET = "/workspace/RAIN/opt_diagonal_net/fixed_benchmark.py"

REPLACEMENTS = [
    # stdlib import needed by the glob-based whole-setting scrub.
    (
        '''import argparse
import base64
import io
''',
        '''import argparse
import base64
import glob
import io
''',
    ),
    # ProblemConfig: carry the setting tag.
    (
        '''    """Immutable descriptor for a sparse-recovery problem setting."""
    dim: int
    sparsity: int
    delta: float = 0.5
''',
        '''    """Immutable descriptor for a sparse-recovery problem setting."""
    dim: int
    sparsity: int
    sigma: float = 0.0
    delta: float = 0.5
''',
    ),
    # Sigma-qualified input key (disjoint files for the two d500 settings).
    (
        '''def _input_key(dim, sparsity, n_max_train, n_test, seed) -> str:
    return (
        f"d{int(dim)}_k{int(sparsity)}"
        f"_nmax{int(n_max_train)}_nt{int(n_test)}_seed{int(seed)}"
    )
''',
        '''def _sigma_tag(sigma) -> str:
    """Filename-safe canonical tag for the --sigma value (e.g. 0.1 -> '0p1')."""
    return f"{float(sigma):g}".replace("-", "m").replace(".", "p")


def _input_key(dim, sparsity, sigma, n_max_train, n_test, seed) -> str:
    return (
        f"d{int(dim)}_k{int(sparsity)}_sig{_sigma_tag(sigma)}"
        f"_nmax{int(n_max_train)}_nt{int(n_test)}_seed{int(seed)}"
    )
''',
    ),
    (
        '''def load_problem(
    dim: int,
    sparsity: int,
    n_max_train: int,
''',
        '''def load_problem(
    dim: int,
    sparsity: int,
    sigma: float,
    n_max_train: int,
''',
    ),
    (
        '''        f"{_input_key(dim, sparsity, n_max_train, n_test, seed)}.npz.b64",
    )
    with open(path, "r") as f:
''',
        '''        f"{_input_key(dim, sparsity, sigma, n_max_train, n_test, seed)}.npz.b64",
    )
    with open(path, "r") as f:
''',
    ),
    # Gate the scrub on MLSBENCH_EPHEMERAL_INPUTS=1 (exported by these eval
    # scripts; apply.py re-materializes the blobs before every evaluation).
    (
        '''def _scrub_inputs(dim, sparsity, n_max_train, n_test, seeds) -> None:
    """Delete the pre-generated input blobs from the workspace once they have
    been loaded into memory, BEFORE any editable hook runs.

    The editable optimizer''',
        '''def _scrub_inputs(dim, sparsity, sigma) -> None:
    """Delete EVERY pre-generated input blob for this (dim, sparsity, sigma)
    setting once the run's datasets are in memory, BEFORE any editable hook
    runs — but only when the harness marks the materialized inputs as
    ephemeral (MLSBENCH_EPHEMERAL_INPUTS=1, i.e. re-created for every
    evaluation). ALL n_max / n_test / seed variants of the setting are removed
    (e.g. the --smoke grid's smaller companion files), not just the variant
    this run loaded: a same-setting variant left on disk would still expose
    held-out targets. Without the flag (no harness staging) blobs persist.

    The editable optimizer''',
    ),
    (
        '''    arrays then live only in this fixed driver's local scope.
    """
    inputs_dir = _inputs_dir()
    for seed in seeds:
        blob = os.path.join(
            inputs_dir,
            f"{_input_key(dim, sparsity, n_max_train, n_test, seed)}.npz.b64",
        )
        try:
            os.remove(blob)
        except OSError:
            pass
''',
        '''    arrays then live only in this fixed driver's local scope.
    """
    if os.environ.get("MLSBENCH_EPHEMERAL_INPUTS") != "1":
        return
    pattern = os.path.join(
        _inputs_dir(),
        f"d{int(dim)}_k{int(sparsity)}_sig{_sigma_tag(sigma)}_*.npz.b64",
    )
    for blob in glob.glob(pattern):
        try:
            os.remove(blob)
        except OSError:
            pass
''',
    ),
    (
        '''    # Load pre-generated clean datasets for all seeds at max training size INTO
    # MEMORY, then DELETE the on-disk blobs BEFORE any editable hook runs. The
    # generator''',
        '''    # Load pre-generated clean datasets for all seeds at max training size INTO
    # MEMORY and, when the harness marks the materialized inputs as ephemeral
    # (MLSBENCH_EPHEMERAL_INPUTS=1: re-created before every evaluation), DELETE
    # the on-disk blobs BEFORE any editable hook runs. The
    # generator''',
    ),
    (
        '''    for seed in seeds:
        datasets[seed] = load_problem(
            problem.dim, problem.sparsity, n_max_train, problem.n_test, seed,
        )
    _scrub_inputs(problem.dim, problem.sparsity, n_max_train, problem.n_test, seeds)
''',
        '''    for seed in seeds:
        datasets[seed] = load_problem(
            problem.dim, problem.sparsity, problem.sigma, n_max_train,
            problem.n_test, seed,
        )
    _scrub_inputs(problem.dim, problem.sparsity, problem.sigma)
''',
    ),
    (
        '''    parser.add_argument("--sigma", type=float, default=0.0,
                        help="(deprecated, ignored) Use --delta instead.")
''',
        '''    parser.add_argument("--sigma", type=float, default=0.0,
                        help="Setting tag: selects this setting's pre-generated "
                             "input files; does not enter the benchmark math "
                             "(training label noise is --delta).")
''',
    ),
    (
        '''    problem = ProblemConfig(
        dim=args.dim,
        sparsity=args.sparsity,
        delta=args.delta,
''',
        '''    problem = ProblemConfig(
        dim=args.dim,
        sparsity=args.sparsity,
        sigma=args.sigma,
        delta=args.delta,
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
} 9>"${_diagnet_patch_lock}"
