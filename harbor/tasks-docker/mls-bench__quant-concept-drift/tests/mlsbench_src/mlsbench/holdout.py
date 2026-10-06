"""Load a task's verifier-side holdout module (``dgp.py``) from a SEALED location.

Why this module exists
----------------------
A-pattern tasks keep the held-out reference -- the generator, the true labels,
the metric -- in ``holdout/<task>/dgp.py``, which is never bind-mounted into the
agent container.  The task's ``parser.py`` imports it at scoring time and grades
the container's emitted predictions against it.

Parsers used to find that module with

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT / "holdout" / "<task>"))
    import dgp

which is correct natively (``<repo>/tasks/<task>/parser.py`` -> ``<repo>``) and
**a scoring leak under Harbor**.  There the parser is exec'd from the verifier's
private meta copy at ``/tmp/mlsbench-verifier.XXXXXX/meta/parser.py``, so
``parents[2]`` is ``/tmp`` and the inserted directory is ``/tmp/holdout/<task>``
-- a fixed, predictable path in a world-writable directory.  The agent phase
runs as root in the same container and ``/tmp`` survives into verification, so a
submission could drop its own ``dgp.py`` there and have the *verifier* import it
and grade the run with it.  Reproduced on the real layout: all-zero predictions
scored a perfect correlation.

The rule this module enforces
-----------------------------
Never derive an import path from ``__file__`` parents, cwd, or an env var in a
parser.  Only two directories are ever consulted, in this order:

1. **The directory the parser itself was loaded from.**  Under Harbor that is
   the verifier's private meta copy -- created with ``mktemp -d`` (so its path
   is not predictable), ``chmod -R a-w``, and sealed 0700 while the submission's
   eval commands run.  The adapter stages the whole ``holdout/<task>/`` tree
   into it, so ``dgp.py`` sits right next to ``parser.py``.  Natively it is
   ``tasks/<task>/`` -- the harness's own read-only task dir.

2. **``<repo>/holdout/<task>/``, and only for a genuine source checkout.**  The
   repo root is taken from the caller, but the fallback is gated on the caller
   actually *being* the repo's task parser: ``<R>/tasks/<task_id>/parser.py``
   with ``<R>/src/mlsbench/__init__.py`` beside it.  Under Harbor the caller
   sits in ``.../meta/``, never in ``<R>/tasks/<task_id>/``, so this candidate
   is not even formed -- whatever an agent plants under ``/tmp``.

The module is then loaded by explicit file path via
``importlib.util.spec_from_file_location``, never by a name that ``sys.path``
gets to resolve.

Sibling imports
---------------
Some holdout modules import a helper that lives beside them (``from tsbad_vus
import ...``), sometimes lazily at scoring time.  The chosen directory -- and
only that one -- is therefore also prepended to ``sys.path``.  That is safe for
exactly the reason above: it is the sealed private meta (or the repo's own
holdout dir), not a guess at a path an agent can write.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

__all__ = ["holdout_dir", "load_holdout_module"]


def holdout_dir(task_id: str, caller_file: str) -> Path:
    """Return the sealed directory holding *task_id*'s holdout modules.

    ``caller_file`` must be the calling module's ``__file__``: the task's
    ``parser.py`` (scoring) or its ``edits/mid_edit.py`` (staging the
    observable inputs, host-side natively and inside the verifier under
    Harbor).  A ``mid_edit`` caller anchors one level up -- ``tasks/<t>/``
    natively, the sealed ``.../meta/`` under Harbor, where the adapter stages
    the whole ``holdout/<t>/`` tree beside ``edits/``.  Raises ``ImportError``
    when no sealed directory carries a ``dgp.py``.
    """
    here = Path(caller_file).resolve().parent
    # An edits/mid_edit.py caller anchors on the directory holding edits/
    # (tasks/<t>/ natively, the sealed meta copy under Harbor).  The parser's
    # own directory is already that anchor.
    anchor = here.parent if here.name == "edits" else here
    candidates = [anchor]

    # Repo fallback -- native runs only.  Gated on the caller genuinely being
    # <R>/tasks/<task_id>/{parser.py,edits/mid_edit.py} inside a source
    # checkout, which the Harbor verifier's meta copy can never satisfy.
    if anchor.name == task_id and len(anchor.parents) >= 2 and anchor.parent.name == "tasks":
        repo = anchor.parents[1]
        if (repo / "src" / "mlsbench" / "__init__.py").is_file():
            candidates.append(repo / "holdout" / task_id)

    for cand in candidates:
        if (cand / "dgp.py").is_file():
            return cand

    raise ImportError(
        f"no sealed holdout directory for {task_id!r}: looked for dgp.py in "
        + ", ".join(str(c) for c in candidates)
        + ". Under Harbor the adapter stages holdout/<task>/ into tests/meta/; "
          "natively it lives at <repo>/holdout/<task>/."
    )


def load_holdout_module(task_id: str, caller_file: str, name: str = "dgp"):
    """Import ``<sealed holdout dir>/<name>.py`` by explicit file path.

    ``caller_file`` must be the calling parser's ``__file__``.  The returned
    module is cached in ``sys.modules`` under a task-scoped key so two tasks
    scored in one process cannot collide, and so a holdout module that pickles
    or re-imports itself still works.
    """
    directory = holdout_dir(task_id, caller_file)
    path = directory / f"{name}.py"
    if not path.is_file():
        raise ImportError(f"{path} does not exist (holdout module {name!r} for {task_id!r})")

    # Let siblings of the holdout module resolve, including the lazy
    # `from <helper> import ...` some of them do at scoring time.  This is the
    # sealed directory, not a predictable agent-writable guess.
    entry = str(directory)
    if entry not in sys.path:
        sys.path.insert(0, entry)

    key = f"_mlsbench_holdout.{task_id.replace('-', '_')}.{name}"
    cached = sys.modules.get(key)
    if cached is not None and getattr(cached, "__file__", None) == str(path):
        return cached

    spec = importlib.util.spec_from_file_location(key, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    # Registered before exec so a self-referential import inside the module
    # resolves to the half-built module instead of re-executing the file.  The
    # key is task-scoped rather than the bare name, so two tasks scored in one
    # process cannot hand each other the wrong dgp.
    sys.modules[key] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(key, None)
        raise
    return module
