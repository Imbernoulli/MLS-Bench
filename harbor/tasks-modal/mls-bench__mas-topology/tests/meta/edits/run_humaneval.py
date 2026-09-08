"""HumanEval benchmark runner for the mas-topology task.

For each HumanEval problem (subset):
1. Format the prompt and call MacNet via run_with_topology.py
2. Extract the generated function from WareHouse output
3. Run unit tests with a timeout
4. Report pass/fail per problem and overall pass@1
"""
import argparse
import ast
import json
import os
import re
import subprocess
import sys
import textwrap
import tempfile
import glob


def load_problems(data_path, subset_step=5):
    """Load HumanEval problems from JSONL, taking every subset_step-th problem."""
    problems = []
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % subset_step == 0:
                problems.append(json.loads(line.strip()))
    return problems


def find_warehouse_dir(name):
    """Find the WareHouse directory for the given project name."""
    warehouse = "./WareHouse"
    if not os.path.exists(warehouse):
        return None
    # Exact match first
    exact = os.path.join(warehouse, name)
    if os.path.isdir(exact):
        return exact
    # Prefix match
    for d in sorted(os.listdir(warehouse)):
        if d.startswith(name):
            return os.path.join(warehouse, d)
    return None


def extract_function_from_files(warehouse_dir, entry_point):
    """Extract function matching entry_point from .py files using AST.

    Returns the raw source text of the function, or None if not found.
    """
    if warehouse_dir is None or not os.path.isdir(warehouse_dir):
        return None

    py_files = glob.glob(os.path.join(warehouse_dir, "*.py"))
    for fpath in py_files:
        try:
            source = open(fpath, "r", encoding="utf-8").read()
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue

        source_lines = source.splitlines(keepends=True)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == entry_point:
                    # Extract source lines for this function
                    start = node.lineno - 1  # 0-indexed
                    end = node.end_lineno  # end_lineno is 1-indexed, inclusive
                    if end is not None:
                        func_source = "".join(source_lines[start:end])
                        return func_source
    return None


def collect_all_py_content(warehouse_dir):
    """Collect all .py file contents from warehouse dir."""
    if warehouse_dir is None or not os.path.isdir(warehouse_dir):
        return ""
    contents = []
    py_files = glob.glob(os.path.join(warehouse_dir, "*.py"))
    for fpath in sorted(py_files):
        try:
            contents.append(open(fpath, "r", encoding="utf-8").read())
        except (UnicodeDecodeError, OSError):
            continue
    return "\n\n".join(contents)


def run_test(prompt, completion, test, entry_point, timeout=10):
    """Run HumanEval test by exec'ing prompt + completion + test + check call.

    Uses a subprocess for isolation and timeout safety.
    Returns True if all tests pass.
    """
    # Build the test program
    test_program = prompt + "\n" + completion + "\n" + test + "\n" + f"check({entry_point})\n"

    # Write to temp file and run in subprocess
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(test_program)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def run_macnet(task_prompt, name, node_num, timeout=300):
    """Call MacNet via run_with_topology.py subprocess.

    Returns True if MacNet completed successfully.
    """
    cmd = [
        sys.executable, "run_with_topology.py",
        "--task", task_prompt,
        "--name", name,
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
            print(f"  MacNet failed (rc={result.returncode}): {stderr}", file=sys.stderr)
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  MacNet timed out after {timeout}s", file=sys.stderr)
        return False
    except Exception as e:
        print(f"  MacNet error: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Run HumanEval benchmark via MacNet")
    parser.add_argument("--node_num", type=int, default=4, help="Number of agent nodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data_path", type=str,
                        default=os.environ.get("HUMANEVAL_PATH", "/root/HumanEval.jsonl"),
                        help="Path to HumanEval JSONL file (env: HUMANEVAL_PATH)")
    parser.add_argument("--subset_step", type=int, default=5,
                        help="Take every N-th problem (default 5 = 33 problems)")
    parser.add_argument("--timeout", type=int, default=10,
                        help="Timeout in seconds for each test execution")
    parser.add_argument("--macnet_timeout", type=int, default=300,
                        help="Timeout in seconds for each MacNet call")
    args = parser.parse_args()

    problems = load_problems(args.data_path, args.subset_step)
    print(f"Loaded {len(problems)} HumanEval problems (every {args.subset_step}th)")
    print(f"Node count: {args.node_num}, Seed: {args.seed}")
    print("=" * 60)

    passed = 0
    total = len(problems)

    for idx, problem in enumerate(problems):
        task_id = problem["task_id"]  # e.g., "HumanEval/0"
        prompt = problem["prompt"]
        test = problem["test"]
        entry_point = problem["entry_point"]

        # Sanitize name for filesystem
        problem_num = task_id.split("/")[-1]
        project_name = f"HumanEval_{problem_num}_{args.seed}"

        print(f"\n[{idx+1}/{total}] {task_id}: {entry_point}")

        # Format task prompt for MacNet
        task_prompt = f"Implement the following Python function:\n\n{prompt}"

        # Run MacNet
        success = run_macnet(task_prompt, project_name, args.node_num,
                             timeout=args.macnet_timeout)

        if not success:
            print(f"PROBLEM {task_id}: FAIL (MacNet error)")
            continue

        # Find warehouse output
        warehouse_dir = find_warehouse_dir(project_name)
        if warehouse_dir is None:
            print(f"PROBLEM {task_id}: FAIL (no output directory)")
            continue

        # Extract the function
        func_source = extract_function_from_files(warehouse_dir, entry_point)

        test_passed = False
        if func_source is not None:
            # Test with extracted function (use it as completion, prompt provides signature)
            test_passed = run_test(prompt, func_source, test, entry_point,
                                   timeout=args.timeout)

        if not test_passed and func_source is None:
            # Fallback: try exec'ing all .py content with the test
            all_content = collect_all_py_content(warehouse_dir)
            if all_content:
                test_passed = run_test("", all_content, test, entry_point,
                                       timeout=args.timeout)

        if test_passed:
            passed += 1
            print(f"PROBLEM {task_id}: PASS")
        else:
            print(f"PROBLEM {task_id}: FAIL")

    # Final metrics
    pass_at_1 = passed / total if total > 0 else 0.0
    print("\n" + "=" * 60)
    print(f"TEST_METRICS pass_at_1={pass_at_1:.4f} passed={passed} total={total}")


if __name__ == "__main__":
    main()
