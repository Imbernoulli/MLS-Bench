#!/usr/bin/env python3
"""Audit rendered Harbor artifacts for all abstractive-summarization siblings."""
from __future__ import annotations

import hashlib
import json
import sys
import tomllib
from pathlib import Path


TASKS = (
    "summ-beam-repetition",
    "summ-beam-width",
    "summ-decoding-length",
    "summ-decoding-temperature",
    "summ-diverse-beam",
    "summ-norepeat-ngram",
    "summ-nucleus-topp",
    "summ-post-truncation",
    "summ-sampling-vs-beam",
    "summ-source-policy",
)
IMAGE = (
    "msai-cn-beijing.cr.volces.com/public/bohanlyu2022/"
    "mlsbench-harbor-abstractive-summarization@"
    "sha256:06b0678dc84d47be4a304a150f9f171e1e37f73fc0788c1fbb5651c0b406497a"
)
VISIBLE_DOMAINS = {
    "summ-beam-repetition": ("[1, 12]", "[0, 20]", "(0, 10]"),
    "summ-beam-width": ("[1, 12]",),
    "summ-decoding-length": ("[0, 200]", "[1, 200]", "(0, 10]"),
    "summ-decoding-temperature": ("[0.05, 5.0]",),
    "summ-diverse-beam": (
        "[1, 12]",
        "[1, num_beams]",
        "exactly 0 when groups == 1",
        "(0, 10]",
    ),
    "summ-norepeat-ngram": ("[0, 20]",),
    "summ-nucleus-topp": ("[0.05, 1.0]",),
    "summ-post-truncation": ("[0, 10000]",),
    "summ-sampling-vs-beam": ("[1, 12]", "(0, 1]", "[0, 1000]", "(0, 5]"),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: audit_summ_rendered.py RENDER_ROOT SOURCE_ROOT")
    render_root = Path(sys.argv[1]).resolve()
    source_root = Path(sys.argv[2]).resolve()
    require(render_root.is_dir(), f"missing render root: {render_root}")

    rendered_tasks = sorted(path.name for path in render_root.glob("summ-*") if path.is_dir())
    require(rendered_tasks == list(TASKS), f"wrong rendered inventory: {rendered_tasks}")

    for task_name in TASKS:
        task = render_root / task_name
        source = source_root / "tasks" / task_name
        required = (
            "environment/Dockerfile",
            "instruction.md",
            "solution/solve.sh",
            "task.toml",
            "tests/test.sh",
            "tests/score_task.py",
            "tests/meta/config.json",
            "tests/meta/leaderboard.csv",
            "tests/meta/parser.py",
            "tests/meta/score_spec.py",
        )
        for relative in required:
            require((task / relative).is_file(), f"{task_name}: missing {relative}")

        dockerfile = (task / "environment" / "Dockerfile").read_text()
        from_lines = [line.strip() for line in dockerfile.splitlines() if line.strip().startswith("FROM ")]
        require(from_lines == [f"FROM {IMAGE}"], f"{task_name}: image is not digest-pinned")
        lower_docker = dockerfile.lower()
        for token in ("pip install", "conda install", "apt-get", "curl ", "wget ", "git clone"):
            require(token not in lower_docker, f"{task_name}: Dockerfile installs at verification time")
        require(
            "rm -rf /data/abstractive-summarization" not in dockerfile,
            f"{task_name}: rendered image deletes verifier data",
        )
        require(
            "RUN rm -rf /workspace/abstractive-summarization" in dockerfile,
            f"{task_name}: stale image-baked package tree remains agent-visible",
        )

        task_toml = tomllib.loads((task / "task.toml").read_text())
        environment = task_toml["environment"]
        require(environment["gpus"] == 1, f"{task_name}: expected exactly one GPU")
        require(environment["gpu_types"] == ["H20"], f"{task_name}: expected H20")
        require(environment["allow_internet"] is False, f"{task_name}: internet must be disabled")

        instruction = (task / "instruction.md").read_text().lower()
        for setting in ("xsum", "cnn/dailymail", "samsum"):
            require(setting in instruction, f"{task_name}: instruction omits {setting}")
        require("512" in instruction, f"{task_name}: instruction omits token cap")
        require("hidden setting" not in instruction, f"{task_name}: hidden-setting semantics leaked")
        require("public setting" not in instruction, f"{task_name}: public-setting semantics leaked")

        config = json.loads((task / "tests" / "meta" / "config.json").read_text())
        require(config["test_cmds"][0]["label"] == task_name, f"{task_name}: wrong command label")
        require(config["seeds"] == [42], f"{task_name}: wrong seed inventory")
        details = config["calibration_protocol_details"]
        require(details["settings_order"] == ["xsum", "cnndm", "samsum"], f"{task_name}: wrong setting order")
        require(details["max_input_tokens"] == 512, f"{task_name}: wrong source-token cap")
        require(details["total_documents"] == 23643, f"{task_name}: wrong document inventory")

        require(
            digest(task / "tests" / "meta" / "parser.py") == digest(source / "parser.py"),
            f"{task_name}: rendered parser drift",
        )
        require(
            digest(task / "tests" / "meta" / "score_spec.py") == digest(source / "score_spec.py"),
            f"{task_name}: rendered score spec drift",
        )
        for relative in config["verifier_only_package_files"]:
            rendered_runtime = (
                task / "tests" / "meta" / "verifier_package_files" / relative
            )
            source_runtime = source_root / "vendor" / relative
            require(
                rendered_runtime.is_file(),
                f"{task_name}: missing verifier-only runtime {relative}",
            )
            require(
                digest(rendered_runtime) == digest(source_runtime),
                f"{task_name}: verifier-only runtime drift for {relative}",
            )
        score_text = (task / "tests" / "meta" / "score_spec.py").read_text()
        for setting in ("xsum", "cnndm", "samsum"):
            require(setting in score_text, f"{task_name}: score omits {setting}")

        eval_script = task / "tests" / "eval" / "scripts" / "run.sh"
        require(eval_script.is_file(), f"{task_name}: missing active eval script")
        require(
            digest(eval_script) == digest(source / "scripts" / "run.sh"),
            f"{task_name}: active eval script drift",
        )
        script = eval_script.read_text()
        require("set -euo pipefail" in script, f"{task_name}: script is not strict")
        require("VERIFICATION_FAILED" in script, f"{task_name}: script lacks failure proof")
        require("CUDA_VISIBLE_DEVICES" not in script, f"{task_name}: script overrides runner GPU")
        for token in ("pip install", "conda install", "apt-get", "curl ", "wget ", "git clone"):
            require(token not in script, f"{task_name}: verification installs or downloads")

        scaffold = task / "environment" / "_scaffold" / "abstractive-summarization"
        require(scaffold.is_dir(), f"{task_name}: missing agent scaffold")
        editable = Path(config["files"][0]["filename"])
        require(
            digest(task / "environment" / "_scaffold" / editable)
            == digest(source / "edits" / "custom_template.py"),
            f"{task_name}: rendered editable scaffold drift",
        )
        agent_visible = instruction + "\n" + "\n".join(
            path.read_text().lower()
            for path in scaffold.rglob("*.py")
            if path.is_file()
        )
        require(
            "mean per-example rouge-l f1" in agent_visible,
            f"{task_name}: exact metric aggregation is not agent-visible",
        )
        require(
            "corpus rouge-l f1" not in agent_visible,
            f"{task_name}: inaccurate corpus-ROUGE wording remains",
        )
        for fragment in VISIBLE_DOMAINS.get(task_name, ()):
            require(
                fragment.lower() in agent_visible,
                f"{task_name}: agent-visible contract omits {fragment!r}",
            )
        leaked = [
            path.relative_to(scaffold).as_posix()
            for path in scaffold.rglob("*")
            if path.is_file()
            and (
                path.name == "common.py"
                or path.name.startswith("harness_")
                or path.name in {"parser.py", "score_spec.py", "leaderboard.csv"}
                or "baseline" in path.parts
                or "anchor" in path.parts
            )
        ]
        require(not leaked, f"{task_name}: verifier or answer material leaked: {leaked}")

    print(
        "SUMM_RENDER_AUDIT tasks=10 parsers=10 score_specs=10 scripts=10 "
        "verifier_runtime=110 scaffolds=10 gpu=1 h20=10 settings=30 "
        "no_leak=10 image_pinned=10"
    )


if __name__ == "__main__":
    main()
