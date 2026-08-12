#!/usr/bin/env python3
"""FIXED evaluation entry for MLS-Bench ephemeral-input tasks.

Problem this solves: the evaluated program is the agent-EDITABLE module, and
Python executes module top-level statements at import time. Launched
directly (``python <editable_module>.py``), any top-level statement the
agent places inside its editable range runs while the staged input blobs
(withheld labels / held-out targets / oracle tables) are still on disk —
BEFORE the module's fixed runner gets a chance to load and scrub them.

This wrapper is FIXED: it lives with the task's eval scripts (native
``tasks/<t>/scripts/``, Harbor ``tests/eval/scripts/``), which are mounted
read-only from outside the agent-editable workspace. It inverts the order:

  1. read every staged blob matching ``--inputs-glob`` into memory;
  2. unlink the blobs (gated on MLSBENCH_EPHEMERAL_INPUTS=1, the harness's
     "inputs are re-staged before every evaluation" marker);
  3. only THEN import the editable module — a hostile top-level statement
     now finds an empty inputs dir;

IMPORTANT — the glob must be scoped to the ACTIVE RUN's blobs only. The
Harbor verifier starts a whole group wave of evaluations CONCURRENTLY
(score_task.py _run_eval_wave), each staging its own (ENV, SEED) blobs into
the shared workspace; a wildcard sweep here would unlink a sibling eval's
just-staged files before that sibling could read them. Each eval script
therefore passes a glob that interpolates its own setting constants and
${SEED} — mirroring the blob naming of the task's mid_edit.py (parity:
{tag}_seed{seed}_s*.labels.b64; diag: the sigma-keyed setting family of
fixed_benchmark._input_key, the same scope as its in-module _scrub_inputs;
nas: the exact nb201_tables_{env}_s{seed}.json). Native runs are strictly
serialized, where exact scoping is simply a further tightening.
  4. inject the in-memory payloads into the FIXED loader module
     (``--inject-module``, default the editable module itself) as
     ``_PRELOADED_INPUTS = {basename: content}``;
  5. call the module's FIXED entry function (``--entry``).

Back-compat: when the module tree predates the injection protocol (the
loader has no ``_PRELOADED_INPUTS`` marker, or the entry function is
missing), the wrapper leaves the blobs on disk and executes the module
unchanged as ``__main__`` — byte-for-byte the legacy flow, whose fixed
runner loads and scrubs in-process.

Accepted residual (documented, not solved here): once the editable module
is imported, agent code shares the interpreter with the fixed runner and
could in principle walk interpreter memory (gc, frames) to reach the loaded
arrays — that in-memory boundary is inherent to the in-process task design
(the fixed runner already holds withheld data in memory while calling agent
hooks). This wrapper's contract is filesystem-only: no withheld bytes
remain ON DISK once any agent-authored code can run.

Usage:
    python fixed_entry.py --module <editable.py> --inputs-glob <glob> \
        [--inputs-glob <glob> ...] [--inject-module <name>] [--entry main] \
        -- [args passed to the module]

Globs are resolved relative to the current working directory (the eval
scripts cd to the workspace root / package dir first).
"""

from __future__ import annotations

import argparse
import glob as _glob
import importlib.util
import os
import runpy
import sys


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    if "--" in argv:
        split = argv.index("--")
        own, rest = argv[:split], argv[split + 1:]
    else:
        own, rest = argv, []
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--module", required=True,
                        help="Path to the agent-editable module to run.")
    parser.add_argument("--inputs-glob", action="append", default=[],
                        help="Glob of staged input blobs to preload+unlink "
                             "(repeatable; relative to the CWD).")
    parser.add_argument("--inject-module", default=None,
                        help="Module name holding the FIXED loaders to inject "
                             "_PRELOADED_INPUTS into (default: the editable "
                             "module itself). Must be importable from the "
                             "editable module's directory.")
    parser.add_argument("--entry", default="main",
                        help="FIXED entry function to invoke (default: main).")
    args = parser.parse_args(own)
    return args, rest


def _read_text(path: str) -> str | None:
    try:
        with open(path, "r") as fh:
            return fh.read()
    except OSError:
        return None


def main() -> int:
    args, module_argv = _parse_args(sys.argv[1:])
    module_path = os.path.abspath(args.module)
    module_dir = os.path.dirname(module_path)
    module_name = os.path.splitext(os.path.basename(module_path))[0]
    inject_name = args.inject_module or module_name

    # ── Protocol detection (BEFORE anything is unlinked) ────────────────
    # The injection protocol requires (a) the FIXED loader module to consult
    # _PRELOADED_INPUTS and (b) the editable module to expose the entry
    # function. Both live in guard-protected fixed regions, so a plain text
    # scan is authoritative; a stale workspace fails it and gets the legacy
    # flow (blobs left on disk, module run as __main__, in-process scrub).
    module_text = _read_text(module_path)
    if module_text is None:
        print(f"[fixed-entry] ERROR: cannot read module {module_path}",
              file=sys.stderr)
        return 2
    if inject_name == module_name:
        inject_text = module_text
    else:
        inject_text = _read_text(os.path.join(module_dir, inject_name + ".py"))
    protocol_ok = (
        inject_text is not None
        and "_PRELOADED_INPUTS" in inject_text
        and f"def {args.entry}(" in module_text
    )

    sys.argv = [module_path] + list(module_argv)
    if module_dir and module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    if not protocol_ok:
        print(
            "[fixed-entry] legacy module (no _PRELOADED_INPUTS protocol); "
            "running it directly on the staged inputs",
            flush=True,
        )
        runpy.run_path(module_path, run_name="__main__")
        return 0

    # ── 1+2: preload the staged blobs, then unlink them ────────────────
    preloaded: dict[str, str] = {}
    ephemeral = os.environ.get("MLSBENCH_EPHEMERAL_INPUTS") == "1"
    for pattern in args.inputs_glob:
        for path in sorted(_glob.glob(pattern)):
            if not os.path.isfile(path):
                continue
            with open(path, "r") as fh:
                preloaded[os.path.basename(path)] = fh.read()
            if ephemeral:
                try:
                    os.remove(path)
                except OSError:
                    pass
    print(
        f"[fixed-entry] preloaded {len(preloaded)} input blob(s)"
        + (" and unlinked them" if ephemeral else " (kept on disk: not "
           "marked ephemeral)")
        + " before importing the editable module",
        flush=True,
    )

    # ── 3: import the editable module (top-level agent code runs NOW, on ──
    # ── an already-empty inputs dir; _PRELOADED_INPUTS is still None)    ──
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # ── 4: inject the payloads into the FIXED loader module ────────────
    target = sys.modules.get(inject_name)
    if target is None:
        # The editable module did not import it itself; do so now (fixed
        # loader modules contain no agent code).
        import importlib as _importlib
        target = _importlib.import_module(inject_name)
    setattr(target, "_PRELOADED_INPUTS", preloaded)

    # ── 5: run the FIXED entry ──────────────────────────────────────────
    getattr(module, args.entry)()
    return 0


if __name__ == "__main__":
    sys.exit(main())
