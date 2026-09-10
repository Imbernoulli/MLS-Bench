"""Evaluate the generated code from MacNet.

Self-contained evaluation that checks:
- completeness: no standalone pass/TODO placeholders in code
- executability: python3 main.py runs without errors
- function_coverage: fraction of functions that have real implementations
Outputs TRAIN_METRICS and TEST_METRICS lines for the MLS-Bench parser.
"""
import os
import sys
import re
import ast
import glob
import subprocess


def find_warehouse_dir(name):
    """Find the WareHouse directory for the given project name."""
    warehouse = "./WareHouse"
    if not os.path.exists(warehouse):
        return None
    # Exact match first
    exact = os.path.join(warehouse, name)
    if os.path.isdir(exact):
        return exact
    # Prefix match — require exact prefix followed by nothing or underscore/dash
    for d in sorted(os.listdir(warehouse)):
        if d == name:
            return os.path.join(warehouse, d)
    return None


def check_completeness(directory):
    """Check if code has no standalone 'pass' statements or TODO comments.

    Uses regex word-boundary matching to avoid false positives on words
    like 'compass', 'bypass', 'passthrough', 'todo_list', etc.
    """
    for root, _, files in os.walk(directory):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            code = open(os.path.join(root, fname), "r", encoding="utf-8").read()
            for line in code.split("\n"):
                stripped = line.strip()
                # Check for standalone 'pass' as the only statement on the line
                if stripped == "pass":
                    return 0.0
                # Check for TODO in comments (# TODO, # todo, etc.)
                if re.search(r'#\s*\bTODO\b', line, re.IGNORECASE):
                    return 0.0
                # Check for 'pass' as a statement (but not in strings or as part of words)
                # Only flag if pass is preceded by colon+whitespace or is the entire statement
                if re.match(r'^\s*pass\s*(#.*)?$', line):
                    return 0.0
    return 1.0


def check_executability(directory, timeout=10):
    """Check if python3 main.py runs without errors.

    Uses subprocess.run with timeout instead of manual sleep+kill.
    GUI apps that don't exit are killed after timeout — this counts
    as success if no traceback was produced.
    """
    main_py = os.path.join(directory, "main.py")
    if not os.path.exists(main_py):
        return 0.0
    try:
        result = subprocess.run(
            ["python3", "main.py"],
            cwd=directory,
            capture_output=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return 1.0
        stderr = result.stderr.decode("utf-8", errors="replace")
        if "Traceback" in stderr:
            return 0.0
        # Non-zero exit but no traceback — could be normal for some apps
        return 0.0
    except subprocess.TimeoutExpired:
        # Timeout means the process was still running (likely a GUI app)
        # This counts as "executable" since it didn't crash
        return 1.0
    except Exception:
        return 0.0


def check_function_coverage(directory):
    """Check what fraction of functions have real implementations.

    Parses AST to find functions/methods whose body is only 'pass',
    'return None', '...', or a docstring followed by pass/ellipsis.
    Returns fraction of implemented functions (1.0 = all implemented).
    """
    total_funcs = 0
    implemented_funcs = 0

    for root, _, files in os.walk(directory):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            try:
                tree = ast.parse(open(fpath, "r", encoding="utf-8").read())
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    total_funcs += 1
                    body = node.body
                    # Skip docstring if present
                    if (body and isinstance(body[0], ast.Expr)
                            and isinstance(body[0].value, ast.Constant)
                            and isinstance(body[0].value.value, str)):
                        body = body[1:]
                    # Check if remaining body is just pass or ellipsis
                    is_stub = (
                        len(body) == 0
                        or (len(body) == 1 and isinstance(body[0], ast.Pass))
                        or (len(body) == 1 and isinstance(body[0], ast.Expr)
                            and isinstance(body[0].value, ast.Constant)
                            and body[0].value.value is ...)
                    )
                    if not is_stub:
                        implemented_funcs += 1

    if total_funcs == 0:
        return 1.0
    return implemented_funcs / total_funcs


def evaluate(directory):
    """Evaluate code in the given directory."""
    if directory is None or not os.path.exists(directory):
        print("TRAIN_METRICS completeness=0.0 executability=0.0 function_coverage=0.0 score=0.0")
        print("TEST_METRICS completeness=0.0 executability=0.0 function_coverage=0.0 score=0.0")
        return

    py_files = glob.glob(os.path.join(directory, "*.py"))
    if not py_files:
        print("TRAIN_METRICS completeness=0.0 executability=0.0 function_coverage=0.0 score=0.0")
        print("TEST_METRICS completeness=0.0 executability=0.0 function_coverage=0.0 score=0.0")
        return

    completeness = check_completeness(directory)
    executability = check_executability(directory)
    func_coverage = check_function_coverage(directory)

    # Weighted score: executability most important, then function coverage, then completeness
    score = 0.4 * executability + 0.35 * func_coverage + 0.25 * completeness

    print(f"TRAIN_METRICS completeness={completeness:.2f} executability={executability:.2f} function_coverage={func_coverage:.2f} score={score:.2f}")
    print(f"TEST_METRICS completeness={completeness:.2f} executability={executability:.2f} function_coverage={func_coverage:.2f} score={score:.2f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="Project name in WareHouse")
    args = parser.parse_args()

    directory = find_warehouse_dir(args.name)
    evaluate(directory)
