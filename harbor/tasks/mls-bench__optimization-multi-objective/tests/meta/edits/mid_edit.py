"""Mid-edit operations for the optimization-multi-objective task.

Two responsibilities:
  1. Create ``deap/custom_moea.py`` — the agent's editable strategy file.
  2. Pre-generate, per (problem, seed), the opaque problem SPEC the agent's
     program loads at run time (numeric configuration + a marshalled black-box
     objective evaluator). The spec carries NO problem name and NO Pareto front.

The benchmark problem identity (ZDT/DTLZ), the analytic true Pareto fronts, and
the IGD / hypervolume / spread metrics live ONLY in
``holdout/optimization-multi-objective/dgp.py`` (host-side, never bind-mounted
into the agent container). Here we write ONLY the opaque spec into the
workspace, keyed by the per-problem alias (``p0``..``p3``) that the run scripts
pass via ``ENV``. The agent never sees which problem is used nor its true
front. The objective evaluation is byte-identical to the original, so honest
results are unchanged.

Producer == consumer interpreter (issue #83): the marshalled code objects in
the specs are loaded by the CONTAINER python (deap image, ``python:3.11-slim``)
but ``marshal`` does not validate the producer's CPython version — a blob
marshalled by a mismatched host interpreter loads silently and then SIGSEGVs
the container on first call. This module therefore marshals the kernels with
the interpreter that will consume them:

  * In-container (Harbor ``tests/eval/_inputgen/apply.py`` runs this module
    inside the task container at eval time): the current interpreter IS the
    consumer, so specs are generated in-process. There is no ``vendor/`` tree
    in that staging layout, which is how the in-container case is detected.
  * Native, host python matches the package ``base_image`` (e.g. host is also
    3.11): in-process generation is already producer == consumer.
  * Native, host python differs: shell out to a container — the task image
    (``vendor/images/deap.sif`` / ``mlsbench/deap:latest``) or any locally
    available ``python:X.Y*`` image matching the base image's version (spec
    generation needs only the stdlib) — running ``dgp.py gen-specs`` with the
    holdout bind-mounted READ-ONLY into that transient helper container. The
    helper is launched by this host-side setup code; the agent never gets a
    shell in it, and the marshalled output carries no problem identity, so
    opacity is preserved.
  * Fallback (no container runtime/image at setup time, e.g. first run on a
    fresh machine before the image is built): best-effort ``docker pull`` /
    ``apptainer pull`` of the small ``python:X.Y-slim`` image; if that also
    fails, fall back to in-process generation. Setup never crashes: every spec
    records its producer magic, and the template's ``_check_spec_magic`` turns
    any residual producer/consumer mismatch into a clear RuntimeError instead
    of a segfault.

Specs are written as base64-encoded JSON text files (the create op writes text),
decoded by ``custom_moea.py``'s ``_load_spec``. The encoded bytes are computed at
setup time, so this file stays small and nothing large is committed.
"""

import base64
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve()
_TASK_DIR = _HERE.parents[1]            # tasks/optimization-multi-objective
_PROJECT_ROOT = _HERE.parents[3]        # repo root
_HOLDOUT_DIR = _PROJECT_ROOT / "holdout" / "optimization-multi-objective"
sys.path.insert(0, str(_HOLDOUT_DIR))
import dgp  # host-only problem defs + fronts + metrics (NOT shipped into the container)

_TEMPLATE_PATH = _HERE.parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

# The runtime seed list is chosen by the harness from the global config and is
# NOT known at workspace-setup time (mid_edit only sees the task config). We
# pre-generate specs for the task's declared seeds plus the standard seed set so
# that whichever seeds the harness runs are always covered.
_STANDARD_SEEDS = [42, 123, 456]
try:
    _cfg = json.loads((_TASK_DIR / "config.json").read_text())
    _SEEDS = sorted(set((_cfg.get("seeds") or []) + _STANDARD_SEEDS))
    # Problem names come from the test_cmd labels (host-side only).
    _PROBLEMS = [e.get("label") for e in _cfg.get("test_cmds", []) if e.get("label")]
except Exception:
    _SEEDS = list(_STANDARD_SEEDS)
    _PROBLEMS = ["zdt1", "zdt3", "dtlz2", "dtlz1"]


def _target_runtime():
    """``(major, minor)`` of the interpreter that will CONSUME the specs, or None.

    Native layout: parsed from the deap package config's ``base_image`` (e.g.
    ``python:3.11-slim``). In the Harbor ``_inputgen`` staging there is no
    ``vendor/`` tree — and there this module is exec'd INSIDE the eval
    container, so the current interpreter is the consumer by construction and
    None (→ in-process generation) is correct.
    """
    cfg = _PROJECT_ROOT / "vendor" / "pkg_configs" / "deap" / "config.json"
    try:
        base_image = json.loads(cfg.read_text()).get("base_image", "")
    except Exception:
        return None
    m = re.search(r"python:(\d+)\.(\d+)", base_image)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _docker_has(image: str) -> bool:
    try:
        r = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True, timeout=60,
        )
        return r.returncode == 0
    except Exception:
        return False


def _run_gen(cmd) -> dict | None:
    """Run a ``dgp.py gen-specs`` container command; parse its stdout JSON."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except Exception as e:
        print(f"[mid_edit] spec generation attempt failed to launch: {e}")
        return None
    if r.returncode != 0:
        tail = (r.stderr or "").strip().splitlines()[-3:]
        print(f"[mid_edit] spec generation attempt exited {r.returncode}: {tail}")
        return None
    if r.stderr:
        print(r.stderr.strip())
    try:
        return json.loads(r.stdout)
    except Exception:
        print("[mid_edit] spec generation produced unparseable stdout")
        return None


def _gen_specs_in_container(target) -> dict | None:
    """Marshal the specs with the CONTAINER interpreter (issue #83).

    Runs ``dgp.py gen-specs`` inside a container whose interpreter matches the
    task image's ``base_image`` version. The holdout dir is bind-mounted
    read-only into the transient helper container; only the opaque specs (no
    problem names, no fronts) come back on stdout. Returns None when no
    suitable container could be used.
    """
    maj, min_ = target
    dgp_container_path = "/holdout_ro/dgp.py"
    gen_args = [
        "python", dgp_container_path, "gen-specs",
        "--problems", ",".join(_PROBLEMS),
        "--seeds", ",".join(str(s) for s in _SEEDS),
    ]
    ro_bind = f"{_HOLDOUT_DIR}:/holdout_ro:ro"

    # Candidate python images whose interpreter matches the consumer. The task
    # image itself is preferred; any python:X.Y* image works because spec
    # generation needs only the stdlib. ``MOEA_SPEC_GEN_IMAGES`` may list
    # extra candidates (e.g. a mirror registry on an offline cluster).
    extra = [i for i in os.environ.get("MOEA_SPEC_GEN_IMAGES", "").split(":") if i]
    docker_images = ["mlsbench/deap:latest"] + extra + [
        f"python:{maj}.{min_}-slim",
        f"python:{maj}.{min_}",
    ]

    if shutil.which("apptainer"):
        sif = _PROJECT_ROOT / "vendor" / "images" / "deap.sif"
        if sif.exists():
            specs = _run_gen(
                ["apptainer", "exec", "--bind", ro_bind, str(sif)] + gen_args
            )
            if specs is not None:
                print(f"[mid_edit] specs marshalled in-container via {sif.name}")
                return specs

    if shutil.which("docker"):
        for image in docker_images:
            if not _docker_has(image):
                continue
            specs = _run_gen(
                ["docker", "run", "--rm", "-v", ro_bind, image] + gen_args
            )
            if specs is not None:
                print(f"[mid_edit] specs marshalled in-container via docker:{image}")
                return specs

    # Nothing suitable locally: best-effort pull of the small python:X.Y-slim
    # image (much lighter than building the task image; setup runs on the host
    # with network, same as `mlsbench build`). Never raises.
    pull_ref = f"python:{maj}.{min_}-slim"
    if shutil.which("docker"):
        try:
            print(f"[mid_edit] no local container image for spec generation; "
                  f"trying docker pull {pull_ref} ...")
            r = subprocess.run(
                ["docker", "pull", pull_ref], capture_output=True, text=True, timeout=600,
            )
            if r.returncode == 0:
                specs = _run_gen(
                    ["docker", "run", "--rm", "-v", ro_bind, pull_ref] + gen_args
                )
                if specs is not None:
                    print(f"[mid_edit] specs marshalled in-container via docker:{pull_ref}")
                    return specs
        except Exception as e:
            print(f"[mid_edit] docker pull fallback failed: {e}")
    elif shutil.which("apptainer"):
        # Pull a throwaway SIF into a shared temp location (atomic via .part +
        # replace, so concurrent baseline setups do not corrupt each other).
        tmp_sif = Path(tempfile.gettempdir()) / f"mlsbench-moo-gen-py{maj}{min_}.sif"
        try:
            if not (tmp_sif.exists() and tmp_sif.stat().st_size > 0):
                part = tmp_sif.with_suffix(f".{os.getpid()}.part")
                print(f"[mid_edit] no local container image for spec generation; "
                      f"trying apptainer pull docker://{pull_ref} ...")
                r = subprocess.run(
                    ["apptainer", "pull", str(part), f"docker://{pull_ref}"],
                    capture_output=True, text=True, timeout=600,
                )
                if r.returncode == 0 and part.exists() and part.stat().st_size > 0:
                    os.replace(part, tmp_sif)
                else:
                    part.unlink(missing_ok=True)
            if tmp_sif.exists() and tmp_sif.stat().st_size > 0:
                specs = _run_gen(
                    ["apptainer", "exec", "--bind", ro_bind, str(tmp_sif)] + gen_args
                )
                if specs is not None:
                    print(f"[mid_edit] specs marshalled in-container via {tmp_sif}")
                    return specs
        except Exception as e:
            print(f"[mid_edit] apptainer pull fallback failed: {e}")

    return None


def _gen_specs_in_process() -> dict:
    return {
        f"{dgp.ALIASES[p]}_seed{s}": dgp.gen_problem(p, seed=s)
        for p in _PROBLEMS
        for s in _SEEDS
    }


def _build_specs() -> dict:
    target = _target_runtime()
    if target is None or sys.version_info[:2] == target:
        # In-container generation (Harbor apply.py), or the host interpreter
        # already matches the task image: in-process marshal IS producer ==
        # consumer.
        return _gen_specs_in_process()
    specs = _gen_specs_in_container(target)
    if specs is not None:
        return specs
    print(
        f"[mid_edit] WARNING: no container interpreter available at setup; "
        f"marshalling specs with the host python "
        f"{sys.version_info.major}.{sys.version_info.minor} instead of "
        f"{target[0]}.{target[1]}. If the eval consumer runs a different "
        "version, the template's producer-magic guard will fail the run with a "
        "clear error (issue #83) — build/pull the deap image (or any matching "
        "python image) and recreate the workspace to fix."
    )
    return _gen_specs_in_process()


def _encode_spec(spec) -> str:
    return base64.b64encode(json.dumps(spec).encode("utf-8")).decode("ascii")


OPS = [
    {
        "op": "create",
        "file": "deap/custom_moea.py",
        "content": _CUSTOM_PY,
    },
]

for _key, _spec in sorted(_build_specs().items()):
    OPS.append({
        "op": "create",
        "file": f"deap/_moea_specs/{_key}.json.b64",
        "content": _encode_spec(_spec),
    })
