#!/usr/bin/env python3
"""In-container input materializer (staged verifier-only at tests/eval/_inputgen/).

Runs at eval time INSIDE the task container — where the generator's deps and
datasets exist — right before the agent program. It imports the task's native
mid_edit.py (staged here in its original relative layout) and writes ONLY the
pre-generated input blobs (never the .py template, which is the agent's editable
file) into the workspace at the exact paths the agent program reads. Because the
generation and the score-phase dgp.truth() both run in this same container with
numpy's version-stable seeded RNG, the inputs and the held-out truth stay
mutually consistent and byte-identical to the native pipeline.
"""
import importlib.util
import json
import os
import sys
import tempfile


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    args = list(sys.argv[1:])
    # --list-out FILE: record the paths this run stages, one per line,
    # INCREMENTALLY (each temp file and destination is appended and flushed
    # BEFORE its bytes are written), so even a mid-run crash or SIGKILL
    # leaves the eval script's EXIT trap with the authoritative superset to
    # delete — never a broad glob, never a sibling eval's files.
    list_out = None
    if "--list-out" in args:
        i = args.index("--list-out")
        list_out = args[i + 1]
        del args[i:i + 2]
    # --emit-json: do NOT write the staged blobs into the workspace at all —
    # print them as a single JSON object {op-file: content} on stdout for the
    # fixed wrapper's --inputs-json-stdin. Harbor runs the whole (label, seed)
    # wave CONCURRENTLY in one shared filesystem, so an on-disk staging window
    # (apply -> wrapper unlink) is readable by a sibling eval's agent code
    # (or a detached watcher left by the pre-wave budget check). Piping the
    # payloads means no withheld bytes ever touch the shared FS (C2). All
    # diagnostics go to stderr in this mode so stdout stays pure JSON.
    emit_json = False
    if "--emit-json" in args:
        args.remove("--emit-json")
        emit_json = True
    task = args[0]
    workspace = args[1] if len(args) > 1 else "/workspace"

    out = sys.stderr if emit_json else sys.stdout

    list_fh = open(list_out, "w") if list_out else None

    def _record(path):
        if list_fh is not None:
            list_fh.write(path + "\n")
            list_fh.flush()

    mid = os.path.join(here, "tasks", task, "edits", "mid_edit.py")
    if not os.path.exists(mid):
        print(f"[inputgen] no mid_edit for {task}; nothing to do", file=out)
        if emit_json:
            print("{}")
        if list_fh is not None:
            list_fh.close()
        return 0
    spec = importlib.util.spec_from_file_location(
        "ig_mid_" + task.replace("-", "_"), mid
    )
    m = importlib.util.module_from_spec(spec)
    # The mid_edit is stdlib-only and silent; the redirect is belt-and-
    # suspenders so a stray print can never corrupt the JSON stream.
    if emit_json:
        sys.stdout = sys.stderr
    try:
        spec.loader.exec_module(m)
    finally:
        if emit_json:
            sys.stdout = sys.__stdout__  # restore real stdout
    n = 0
    staged = []
    emitted = {}
    try:
        for op in getattr(m, "OPS", []):
            if op.get("op") != "create":
                continue
            f = op.get("file", "")
            if f.endswith(".py"):
                continue  # never overwrite the agent's editable program
            if emit_json:
                # Key by the op's workspace-relative file; the wrapper keys
                # its preload map by basename, exactly as the disk flow does.
                emitted[f] = op.get("content", "")
                n += 1
                continue
            dst = os.path.join(workspace, f)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            # Atomic write: group evals run concurrently and each materializes
            # the SAME shared input files, so a plain truncating open() can be
            # observed mid-write by another eval. Write to a temp in the same
            # dir + os.replace (atomic on POSIX) so a concurrent reader always
            # sees a complete file. The temp is created empty, then recorded
            # (with the destination) BEFORE the secret bytes are written.
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dst) or ".", suffix=".tmp")
            _record(tmp)
            _record(dst)
            try:
                with os.fdopen(fd, "w") as fh:
                    fh.write(op.get("content", ""))
                os.replace(tmp, dst)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
            n += 1
            staged.append(dst)
        if emit_json:
            json.dump(emitted, sys.__stdout__)
            sys.__stdout__.flush()
            print(file=sys.__stdout__)
            print(f"[inputgen] emitted {n} input blob(s) for {task} "
                  "(no on-disk staging)", file=sys.stderr)
        else:
            print(f"[inputgen] materialized {n} input file(s) for {task}")
        return 0
    except BaseException:
        # Belt and braces: remove everything staged so far before exiting
        # non-zero (the eval script's EXIT trap also consumes the partial
        # --list-out record, which covers SIGKILL where this never runs).
        for dst in staged:
            try:
                os.unlink(dst)
            except OSError:
                pass
        raise
    finally:
        if list_fh is not None:
            list_fh.close()


if __name__ == "__main__":
    sys.exit(main())
