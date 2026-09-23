#!/usr/bin/env python3
"""Solvable Pass Rate (SoPR) judge for agent-tool-reasoning.

Runs StableToolBench's answer judge (toolbench/tooleval, evaluator
``tooleval_gpt-3.5-turbo_default`` = ReinforceToolLearningEvaluator) on the
answer files of ONE test invocation and prints

    TEST_METRICS: sopr=<x> sopr_n_scored=<n>

train.sh calls it right after inference, so every eval row carries SoPR;
compute_sopr.sh calls it to backfill old rows. Both go through this file, so
eval-time and backfilled values are computed the same way.

This is eval_pass_rate.py's ``compute_pass_rate`` + scoring loop (the path
compute_sopr.sh used to produce the leaderboard's sopr_* values) with three
differences, none of which changes an honest run's value:
  * Inputs: only ``<answer_dir>/<qid>_CustomSearch.json`` for the query ids in
    ``--query_file`` are read, never other files in the directory.
  * Denominator: every query in ``--query_file``. A query whose answer file is
    missing or cannot be converted scores 0 (upstream dropped it from the
    denominator, so a crashing policy could raise its SoPR). The leaderboard
    baselines have n_scored = 50 = the query count, so their values are the
    same either way.
  * Transport: the judge endpoint/key are set on the evaluator directly
    (OpenRouter, key from $OPENROUTER_API_KEY_NEW) instead of through an
    api_pool.json file, and a failed judge call is retried for longer than
    upstream's 15 s before the run fails. A judge that still fails aborts
    the run with a non-zero exit; it never scores the query silently.
"""
import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

JUDGE_API_BASE = "https://openrouter.ai/api/v1"
JUDGE_KEY_ENV = "OPENROUTER_API_KEY_NEW"
EVALUATOR = "tooleval_gpt-3.5-turbo_default"
PARSER_METHOD = "DFS_CustomSearch"  # CustomSearch writes a tree -> upstream DFS branch


def _default_pkg_root() -> str:
    env = os.environ.get("MLSBENCH_PKG_DIR", "")
    if env and os.path.isdir(os.path.join(env, "toolbench", "tooleval")):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "..", "vendor",
                                        "external_packages", "stabletoolbench"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--answer_dir", required=True)
    p.add_argument("--query_file", required=True)
    p.add_argument("--pkg_root", default=None,
                   help="stabletoolbench package root (holds toolbench/tooleval)")
    p.add_argument("--eval_model", default="meta-llama/llama-3.3-70b-instruct")
    p.add_argument("--evaluate_times", type=int, default=1)
    p.add_argument("--max_eval_threads", type=int, default=4)
    p.add_argument("--max_attempts", type=int, default=8,
                   help="attempts per judge call before the run fails")
    p.add_argument("--save_path", default="",
                   help="optional file for the per-query verdicts (JSON)")
    args = p.parse_args()

    key = os.environ.get(JUDGE_KEY_ENV, "").strip()
    if not key:
        print(f"ERROR: SoPR judge key missing: set {JUDGE_KEY_ENV} (OpenRouter). "
              f"Without it SoPR cannot be computed and this setting scores 0.",
              file=sys.stderr, flush=True)
        return 2

    pkg_root = os.path.abspath(args.pkg_root or _default_pkg_root())
    tooleval_dir = os.path.join(pkg_root, "toolbench", "tooleval")
    if not os.path.isdir(os.path.join(tooleval_dir, "evaluators", EVALUATOR)):
        print(f"ERROR: StableToolBench judge not found under {tooleval_dir}",
              file=sys.stderr, flush=True)
        return 2
    sys.path.insert(0, tooleval_dir)
    # The evaluator reads the model name from EVAL_MODEL (see
    # evaluators/registered_cls/tooleval.py::function_call).
    os.environ["EVAL_MODEL"] = args.eval_model

    from convert_to_answer_format import process_invalid_data, process_valid_data  # noqa: E402
    from evaluators import load_registered_automatic_evaluator  # noqa: E402
    from evaluators.registered_cls.rtl import AnswerStatus  # noqa: E402
    from utils import get_steps  # noqa: E402

    with open(args.query_file) as fh:
        query_ids = [str(q["query_id"]) for q in json.load(fh)]
    if not query_ids or len(set(query_ids)) != len(query_ids):
        print(f"ERROR: bad query file {args.query_file}", file=sys.stderr, flush=True)
        return 2

    # ── convert (convert_answers_local.py logic, restricted to the fixed ids) ──
    examples: dict[str, dict] = {}
    missing: list[str] = []
    unconvertible: list[str] = []
    for qid in query_ids:
        path = os.path.join(args.answer_dir, f"{qid}_CustomSearch.json")
        if not os.path.isfile(path):
            missing.append(qid)
            continue
        try:
            with open(path) as fh:
                data_dict = json.load(fh)
            if not data_dict["answer_generation"]["valid_data"]:
                examples[qid] = process_invalid_data(PARSER_METHOD, data_dict)
            else:
                examples[qid] = process_valid_data(PARSER_METHOD, data_dict["answer_generation"])
        except Exception as exc:  # noqa: BLE001
            unconvertible.append(qid)
            print(f"WARN: failed to convert {path}: {exc!r}", file=sys.stderr, flush=True)
    if missing:
        print(f"WARNING: SoPR: {len(missing)}/{len(query_ids)} queries have no answer "
              f"file; they score 0", file=sys.stderr, flush=True)
    if unconvertible:
        print(f"WARNING: SoPR: {len(unconvertible)}/{len(query_ids)} answer files could "
              f"not be converted; they score 0", file=sys.stderr, flush=True)

    # ── judge (eval_pass_rate.py::compute_pass_rate) ──
    evaluators = []
    for _ in range(args.max_eval_threads):
        ev = load_registered_automatic_evaluator(
            evaluator_name=EVALUATOR,
            evaluators_cfg_path=os.path.join(tooleval_dir, "evaluators"))
        ev.opr.pool = [{"api_key": key, "api_base": JUDGE_API_BASE}]
        ev.opr.now_pos = 0
        evaluators.append(ev)

    def judge(qid: str, evaluate_time: int):
        example = examples[qid]
        _answer_steps, final_step = get_steps(example)
        if "'name': 'Finish'" not in final_step:
            return qid, evaluate_time, AnswerStatus.Unsolved
        delay = 1.0
        for attempt in range(1, args.max_attempts + 1):
            try:
                is_solved, _reason = random.choice(evaluators).check_is_solved(
                    {"query": example["query"], "available_tools": example["available_tools"]},
                    example["answer"],
                    return_reason=True,
                )
                return qid, evaluate_time, is_solved
            except Exception as exc:  # noqa: BLE001
                if attempt == args.max_attempts:
                    raise RuntimeError(
                        f"SoPR judge failed for query {qid} after {attempt} attempts: {exc!r}"
                    ) from exc
                print(f"WARN: judge call for query {qid} failed (attempt {attempt}): "
                      f"{exc!r}; retrying", file=sys.stderr, flush=True)
                time.sleep(delay * (0.5 + random.random()))
                delay = min(delay * 2, 60.0)

    verdicts: dict[str, dict[int, str]] = {qid: {} for qid in examples}
    jobs = [(qid, t) for qid in query_ids if qid in examples for t in range(args.evaluate_times)]
    try:
        with ThreadPoolExecutor(args.max_eval_threads) as pool:
            for qid, t, status in pool.map(lambda j: judge(*j), jobs):
                verdicts[qid][t] = str(status)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1

    # ── score (eval_pass_rate.py scoring loop; denominator = all query ids) ──
    scores = []
    for t in range(args.evaluate_times):
        score = 0.0
        for qid in verdicts:
            v = verdicts[qid].get(t)
            if v == "AnswerStatus.Solved":
                score += 1.0
            elif v == "AnswerStatus.Unsure":
                score += 0.5
        scores.append(score / len(query_ids))
    sopr = sum(scores) / len(scores)

    if args.save_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
        with open(args.save_path, "w") as fh:
            json.dump({"eval_model": args.eval_model, "evaluator": EVALUATOR,
                       "evaluate_times": args.evaluate_times, "n_total": len(query_ids),
                       "missing": missing, "unconvertible": unconvertible,
                       "verdicts": verdicts, "sopr": sopr}, fh, indent=2)

    print(f"TEST_METRICS: sopr={sopr:.4f} sopr_n_scored={len(examples)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
