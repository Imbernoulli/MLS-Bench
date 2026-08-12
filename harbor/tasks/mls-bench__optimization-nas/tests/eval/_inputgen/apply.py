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
import os
import sys
import tempfile


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    args = list(sys.argv[1:])
    # --list-out FILE: record the absolute path of every staged file, one per
    # line. The eval script's EXIT trap uses it as the exact scope for its
    # crash backstop (delete THIS eval's staged blobs even when the runner
    # died before its own in-process scrub) — never a broad glob.
    list_out = None
    if "--list-out" in args:
        i = args.index("--list-out")
        list_out = args[i + 1]
        del args[i:i + 2]
    task = args[0]
    workspace = args[1] if len(args) > 1 else "/workspace"

    def _write_list(paths):
        if list_out:
            with open(list_out, "w") as fh:
                fh.write("".join(p + "\n" for p in paths))

    mid = os.path.join(here, "tasks", task, "edits", "mid_edit.py")
    if not os.path.exists(mid):
        print(f"[inputgen] no mid_edit for {task}; nothing to do")
        _write_list([])
        return 0
    spec = importlib.util.spec_from_file_location(
        "ig_mid_" + task.replace("-", "_"), mid
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    n = 0
    staged = []
    for op in getattr(m, "OPS", []):
        if op.get("op") != "create":
            continue
        f = op.get("file", "")
        if f.endswith(".py"):
            continue  # never overwrite the agent's editable program
        dst = os.path.join(workspace, f)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        # Atomic write: group evals run concurrently and each materializes the
        # SAME shared input files, so a plain truncating open() can be observed
        # mid-write by another eval. Write to a temp in the same dir + os.replace
        # (atomic on POSIX) so a concurrent reader always sees a complete file.
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dst) or ".", suffix=".tmp")
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
    _write_list(staged)
    print(f"[inputgen] materialized {n} input file(s) for {task}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
