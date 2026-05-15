"""SRDD (Software Requirement Document Dataset) benchmark runner for mas-topology.

For each SRDD query prompt:
1. Call MacNet via run_with_topology.py with --task <prompt> --name <id> --type <category>
2. Locate the generated WareHouse/<name>/ output directory
3. Run an executability check on the generated project
4. Report per-query PASS/FAIL and overall executability rate

== Executability Check Rules (frozen — do not change without re-running all baselines) ==

Entry-point resolution order:
  1. main.py in WareHouse/<name>/
  2. If main.py does not exist, the single .py file (if exactly one .py file exists)
  3. If multiple .py files but no main.py, FAIL (ambiguous entry point)
  4. If no .py files at all, FAIL (no code generated)

Execution:
  - Command: python3 <entry_point>
  - Working directory: WareHouse/<name>/
  - stdin: /dev/null (EOF on any input() call — prevents hangs)
  - Timeout: 10 seconds

PASS conditions (any of):
  - Process exits with code 0 within timeout
  - Process is still running at timeout (killed by SIGTERM) AND stderr does
    not contain "Traceback" — this covers GUI/server apps that start successfully

FAIL conditions:
  - No WareHouse/<name>/ directory produced
  - No valid entry point found (rules above)
  - Process exits with code != 0 within timeout
  - Process exits with code 0 but stderr contains "Traceback"
  - Process still running at timeout AND stderr contains "Traceback"

Metrics emitted:
  TEST_METRICS srdd_exec_rate=<float> total=<int> passed=<int> mean_loc=<float>
  - srdd_exec_rate: passed / total
  - total: number of queries attempted
  - passed: number that met PASS conditions
  - mean_loc: average lines of Python code across all generated projects
"""
import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import time


def load_queries(query_path, subset_step=1):
    """Load SRDD queries from JSON file, taking every subset_step-th query."""
    with open(query_path, "r", encoding="utf-8") as f:
        queries = json.loads(f.read())
    return [q for i, q in enumerate(queries) if i % subset_step == 0]


def find_warehouse_dir(name):
    """Find the WareHouse directory for the given project name.

    MacNet creates WareHouse/<name>/ (exact match to --name argument).
    Falls back to prefix match in case of unexpected suffixes.
    """
    warehouse = "./WareHouse"
    if not os.path.exists(warehouse):
        return None
    # Exact match first
    exact = os.path.join(warehouse, name)
    if os.path.isdir(exact):
        return exact
    # Prefix match fallback
    for d in sorted(os.listdir(warehouse)):
        if d.startswith(name):
            return os.path.join(warehouse, d)
    return None


def find_entry_point(warehouse_dir):
    """Resolve the entry-point .py file.

    Returns (path, reason) where reason is None on success or an error string.
    """
    if warehouse_dir is None or not os.path.isdir(warehouse_dir):
        return None, "no output directory"

    main_py = os.path.join(warehouse_dir, "main.py")
    if os.path.isfile(main_py):
        return main_py, None

    py_files = [f for f in glob.glob(os.path.join(warehouse_dir, "*.py"))
                if os.path.isfile(f)]
    if len(py_files) == 1:
        return py_files[0], None
    if len(py_files) == 0:
        return None, "no .py files generated"
    return None, f"multiple .py files but no main.py ({len(py_files)} files)"


def count_loc(warehouse_dir):
    """Count total lines of Python code in the warehouse directory."""
    if warehouse_dir is None or not os.path.isdir(warehouse_dir):
        return 0
    total = 0
    for fpath in glob.glob(os.path.join(warehouse_dir, "*.py")):
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                total += sum(1 for line in f if line.strip())
        except OSError:
            continue
    return total


def check_executability(entry_point, warehouse_dir, timeout=10):
    """Run the entry point and determine PASS/FAIL per the frozen rules.

    Returns (passed: bool, reason: str).
    """
    try:
        proc = subprocess.Popen(
            [sys.executable, os.path.basename(entry_point)],
            cwd=warehouse_dir,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,
        )
    except Exception as e:
        return False, f"failed to start: {e}"

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        stderr_text = stderr.decode("utf-8", errors="replace")
        has_traceback = "traceback" in stderr_text.lower()

        if proc.returncode == 0 and not has_traceback:
            return True, "exit 0"
        elif proc.returncode == 0 and has_traceback:
            return False, "exit 0 but stderr has Traceback"
        else:
            return False, f"exit {proc.returncode}"
    except subprocess.TimeoutExpired:
        # Process still running — kill it and check stderr
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except OSError:
            pass
        time.sleep(0.5)
        try:
            proc.kill()
        except OSError:
            pass
        stderr_text = ""
        try:
            _, stderr_bytes = proc.communicate(timeout=3)
            stderr_text = stderr_bytes.decode("utf-8", errors="replace")
        except Exception:
            pass

        has_traceback = "traceback" in stderr_text.lower()
        if has_traceback:
            return False, "timeout + Traceback in stderr"
        return True, "still running at timeout (killed) — assumed GUI/server"


def run_macnet(task_prompt, name, sw_type, node_num, timeout=300):
    """Call MacNet via run_with_topology.py subprocess.

    Returns True if MacNet completed successfully.
    """
    cmd = [
        sys.executable, "run_with_topology.py",
        "--task", task_prompt,
        "--name", name,
        "--type", sw_type,
        "--node_num", str(node_num),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            env={**os.environ, "NODE_NUM": str(node_num)},
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")[-500:]
            print(f"  MacNet failed (rc={result.returncode}): {stderr}",
                  file=sys.stderr)
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  MacNet timed out after {timeout}s", file=sys.stderr)
        return False
    except Exception as e:
        print(f"  MacNet error: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run SRDD benchmark via MacNet")
    parser.add_argument("--node_num", type=int, default=4,
                        help="Number of agent nodes")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (for reproducibility logging)")
    parser.add_argument("--query_path", type=str,
                        default=os.environ.get("SRDD_QUERY_PATH",
                                               "srdd_queries.json"),
                        help="Path to SRDD query JSON (env: SRDD_QUERY_PATH)")
    parser.add_argument("--subset_step", type=int, default=1,
                        help="Take every N-th query (default 1 = all)")
    parser.add_argument("--exec_timeout", type=int, default=10,
                        help="Timeout for executability check (seconds)")
    parser.add_argument("--macnet_timeout", type=int, default=300,
                        help="Timeout for each MacNet call (seconds)")
    args = parser.parse_args()

    queries = load_queries(args.query_path, args.subset_step)
    print(f"Loaded {len(queries)} SRDD queries (every {args.subset_step}th)")
    print(f"Node count: {args.node_num}, Seed: {args.seed}")
    print("=" * 60)

    passed = 0
    total = len(queries)
    loc_sum = 0

    for idx, query in enumerate(queries):
        qid = query["id"]
        sw_type = query["type"]
        task_prompt = query["task"]
        project_name = f"{qid}_{args.seed}"

        print(f"\n[{idx+1}/{total}] {qid} ({sw_type})")

        # Run MacNet
        success = run_macnet(task_prompt, project_name, sw_type,
                             args.node_num, timeout=args.macnet_timeout)
        if not success:
            print(f"QUERY {qid}: FAIL (MacNet error)")
            continue

        # Find warehouse output
        warehouse_dir = find_warehouse_dir(project_name)
        if warehouse_dir is None:
            print(f"QUERY {qid}: FAIL (no output directory)")
            continue

        # Count LOC
        loc = count_loc(warehouse_dir)
        loc_sum += loc

        # Find entry point
        entry_point, err = find_entry_point(warehouse_dir)
        if entry_point is None:
            print(f"QUERY {qid}: FAIL ({err}), LOC={loc}")
            continue

        # Executability check
        ok, reason = check_executability(entry_point, warehouse_dir,
                                         timeout=args.exec_timeout)
        if ok:
            passed += 1
            print(f"QUERY {qid}: PASS ({reason}), LOC={loc}")
        else:
            print(f"QUERY {qid}: FAIL ({reason}), LOC={loc}")

    # Final metrics
    exec_rate = passed / total if total > 0 else 0.0
    mean_loc = loc_sum / total if total > 0 else 0.0
    print("\n" + "=" * 60)
    print(f"TEST_METRICS srdd_exec_rate={exec_rate:.4f} total={total} "
          f"passed={passed} mean_loc={mean_loc:.1f}")


if __name__ == "__main__":
    main()
