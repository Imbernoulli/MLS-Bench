"""Locate and validate the editable DLMRefreshPolicy block."""

import json
from pathlib import Path


def policy_span() -> tuple[int, int]:
    task_dir = Path(__file__).resolve().parents[1]
    lines = (task_dir / "edits" / "custom_template.py").read_text().splitlines()
    start = next(
        i for i, line in enumerate(lines, start=1)
        if line.startswith("class DLMRefreshPolicy:")
    )
    marker = next(
        i for i, line in enumerate(lines, start=1)
        if line.startswith("# end of editable region")
    )
    span = (start, marker - 1)

    config = json.loads((task_dir / "config.json").read_text())
    configured = None
    for entry in config.get("files", []):
        if entry.get("filename") == "dLLM-cache/custom_dlm_eval.py":
            edits = entry.get("edit") or []
            if edits:
                configured = (int(edits[0]["start"]), int(edits[0]["end"]))
                break
    if configured != span:
        raise RuntimeError(
            "DLMRefreshPolicy edit span drifted: "
            f"custom_template.py has {span}, but config.json has {configured}. "
            "Update config.json before running baselines."
        )
    return span
