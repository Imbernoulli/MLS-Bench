#!/usr/bin/env python3
"""Convert raw qa_pipeline_multithread JSON outputs into the format
expected by stabletoolbench's eval_pass_rate.py.

The upstream convert_to_answer_format.py uses a single ``--method`` arg
both as a filename substring filter AND as a parser branch selector
('DFS' substring picks the tree parser, 'CoT' picks the linear parser).
Our raw files are named ``<qid>_CustomSearch.json``, which doesn't
contain 'DFS' or 'CoT', so the upstream script either skips them or
raises NotImplementedError.

This wrapper iterates files matching ``--filename_substring`` (default
'CustomSearch') and parses them with the DFS branch (since CustomSearch
uses a Tree internally for greedy_chain / dfs_ranked / dfsdt alike).
"""
import argparse
import json
import os
import sys

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "vendor", "external_packages", "stabletoolbench"))
sys.path.insert(0, os.path.join(PKG_ROOT, "toolbench", "tooleval"))

from convert_to_answer_format import process_valid_data, process_invalid_data  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--answer_dir", required=True)
    p.add_argument("--filename_substring", default="CustomSearch")
    p.add_argument("--parser_method", default="DFS_CustomSearch",
                   help="Must contain 'DFS' or 'CoT' to pick a valid upstream parser branch.")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    answer_dict: dict[str, dict] = {}
    for filename in sorted(os.listdir(args.answer_dir)):
        if not filename.endswith(".json"):
            continue
        if args.filename_substring not in filename:
            continue
        qid = filename.split("_")[0]
        with open(os.path.join(args.answer_dir, filename)) as f:
            data_dict = json.load(f)
        try:
            if not data_dict["answer_generation"]["valid_data"]:
                answer_dict[qid] = process_invalid_data(args.parser_method, data_dict)
            else:
                answer_dict[qid] = process_valid_data(args.parser_method, data_dict["answer_generation"])
        except Exception as exc:  # noqa: BLE001
            print(f"WARN: failed to convert {filename}: {exc}", file=sys.stderr)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(answer_dict, f, indent=2)
    print(f"Converted {len(answer_dict)} answers -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
