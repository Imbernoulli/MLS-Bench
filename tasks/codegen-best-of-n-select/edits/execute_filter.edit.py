"""Unmeasured candidate: rank with PROVIDED-test execution results.

Run each candidate against the PROVIDED example tests (safely, via
common.passes_all) and submit the first candidate that passes all of them; if
none pass, choose the candidate passing the most assertions, with stable ties.
Reference surface: vendor/code-generation-lab/solution/policy_select.py
"""

_FILE = "code-generation-lab/solution/policy_select.py"

_CONTENT = '''def select_candidate(candidates, problem, tok):
    vis = problem["visible_tests"]
    setup = problem.get("test_setup", "")
    best_i, best_passed = 0, -1
    for i, c in enumerate(candidates):
        r = common.run_tests(c, vis, setup)
        if r["ok"]:
            return i
        if r["passed"] > best_passed:
            best_i, best_passed = i, r["passed"]
    return best_i'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 25, "end_line": 28, "content": _CONTENT},
]
