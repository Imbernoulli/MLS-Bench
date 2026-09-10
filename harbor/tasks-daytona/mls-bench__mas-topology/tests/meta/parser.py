"""Task-specific output parser for mas-topology.

Dispatches on cmd_label (3 settings):

  humaneval-4-deepseek:
    PROBLEM HumanEval/N: PASS or FAIL  (per-problem feedback)
    TEST_METRICS pass_at_1=X.XX passed=N total=M
    Leaderboard metrics: pass_at_1_deepseek, passed_deepseek, total_deepseek

  humaneval-4-qwen:
    Same format as above, metric suffix "_qwen"

  srdd-4-deepseek:
    QUERY srdd_NNN: PASS or FAIL  (per-query feedback)
    TEST_METRICS srdd_exec_rate=X.XX total=N passed=M mean_loc=Y.Y
    Leaderboard metrics: srdd_exec_rate, passed, mean_loc
"""

import re
import sys
from pathlib import Path

# Allow importing from mlsbench package when run standalone
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the mas-topology (multi-agent topology) task."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        if cmd_label.startswith("srdd"):
            return self._parse_srdd(raw_output)
        # humaneval-4-deepseek or humaneval-4-qwen
        if cmd_label.endswith("-qwen"):
            suffix = "_qwen"
        elif cmd_label.endswith("-deepseek"):
            suffix = "_deepseek"
        else:
            suffix = ""
        return self._parse_humaneval(raw_output, suffix)

    def _parse_humaneval(self, raw_output: str, suffix: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        problem_lines = []
        for line in raw_output.splitlines():
            if line.strip().startswith("PROBLEM HumanEval/"):
                problem_lines.append(line.strip())

        if problem_lines:
            n_pass = sum(1 for l in problem_lines if "PASS" in l)
            n_fail = sum(1 for l in problem_lines if "FAIL" in l)
            feedback_parts.append(
                f"HumanEval results: {n_pass} passed, {n_fail} failed out of {len(problem_lines)} problems"
            )
            failed = [l for l in problem_lines if "FAIL" in l]
            if failed:
                feedback_parts.append(
                    "Failed problems:\n" + "\n".join(failed[:20])
                )

        for line in raw_output.splitlines():
            match = re.search(
                r"TEST_METRICS\s+pass_at_1=([\d.]+)\s+passed=(\d+)\s+total=(\d+)",
                line,
            )
            if match:
                metrics[f"pass_at_1{suffix}"] = float(match.group(1))
                metrics[f"passed{suffix}"] = int(match.group(2))
                metrics[f"total{suffix}"] = int(match.group(3))
                feedback_parts.append(line.strip())

        if feedback_parts:
            feedback = "\n\n".join(feedback_parts)
        else:
            lines = raw_output.strip().splitlines()
            feedback = "\n".join(lines[-50:])

        return ParseResult(feedback=feedback, metrics=metrics)

    def _parse_srdd(self, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        query_lines = []
        for line in raw_output.splitlines():
            if line.strip().startswith("QUERY srdd_"):
                query_lines.append(line.strip())

        if query_lines:
            n_pass = sum(1 for l in query_lines if ": PASS" in l)
            n_fail = sum(1 for l in query_lines if ": FAIL" in l)
            feedback_parts.append(
                f"SRDD results: {n_pass} passed, {n_fail} failed out of {len(query_lines)} queries"
            )
            failed = [l for l in query_lines if ": FAIL" in l]
            if failed:
                feedback_parts.append(
                    "Failed queries:\n" + "\n".join(failed[:20])
                )

        for line in raw_output.splitlines():
            match = re.search(
                r"TEST_METRICS\s+srdd_exec_rate=([\d.]+)\s+total=(\d+)\s+"
                r"passed=(\d+)\s+mean_loc=([\d.]+)",
                line,
            )
            if match:
                metrics["srdd_exec_rate"] = float(match.group(1))
                metrics["srdd_total"] = int(match.group(2))
                metrics["srdd_passed"] = int(match.group(3))
                metrics["mean_loc"] = float(match.group(4))
                feedback_parts.append(line.strip())

        if feedback_parts:
            feedback = "\n\n".join(feedback_parts)
        else:
            lines = raw_output.strip().splitlines()
            feedback = "\n".join(lines[-50:])

        return ParseResult(feedback=feedback, metrics=metrics)
