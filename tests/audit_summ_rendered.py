#!/usr/bin/env python3
"""Audit rendered Harbor artifacts for all abstractive-summarization siblings."""
from __future__ import annotations

import hashlib
import json
import runpy
import stat
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
PACKAGE = "abstractive-summarization"
NATIVE_BASELINES = {
    "summ-beam-repetition": "greedy",
    "summ-beam-width": "greedy",
    "summ-decoding-length": "short",
    "summ-decoding-temperature": "hot",
    "summ-diverse-beam": "diverse",
    "summ-norepeat-ngram": "off",
    "summ-nucleus-topp": "wide",
    "summ-post-truncation": "one",
    "summ-sampling-vs-beam": "sample",
    "summ-source-policy": "empty",
}
VERIFIER_RUNTIME = {
    "abstractive-summarization/common.py",
    "abstractive-summarization/harness_beam.py",
    "abstractive-summarization/harness_beamwidth.py",
    "abstractive-summarization/harness_diverse.py",
    "abstractive-summarization/harness_length.py",
    "abstractive-summarization/harness_norepeat.py",
    "abstractive-summarization/harness_posttrunc.py",
    "abstractive-summarization/harness_sampling.py",
    "abstractive-summarization/harness_source.py",
    "abstractive-summarization/harness_temperature.py",
    "abstractive-summarization/harness_topp.py",
}
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


def _require_source_config_contract(
    task_name: str,
    rendered_config: dict,
    source_config: dict,
    all_editable_files: set[str],
) -> None:
    for key, value in source_config.items():
        require(
            key in rendered_config and rendered_config[key] == value,
            f"{task_name}: rendered config drift for source key {key!r}",
        )
    require(
        set(rendered_config) - set(source_config)
        == {"agent_pruned_package_files"},
        f"{task_name}: unexpected rendered-only config keys",
    )
    current_editable = source_config["files"][0]["filename"]
    require(
        rendered_config["agent_pruned_package_files"]
        == sorted(all_editable_files - {current_editable}),
        f"{task_name}: wrong sibling-solution pruning inventory",
    )


def _docker_instructions(dockerfile: str) -> list[str]:
    instructions = []
    current = ""
    for raw_line in dockerfile.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        current = f"{current} {line}".strip()
        if current.endswith("\\"):
            current = current[:-1].rstrip()
            continue
        instructions.append(current)
        current = ""
    require(not current, "Dockerfile ends with an unterminated continuation")
    return instructions


def _require_dockerfile_contract(task_name: str, dockerfile: str) -> None:
    expected = [
        f"FROM {IMAGE}",
        "RUN rm -rf /workspace/abstractive-summarization",
        "COPY _scaffold/ /workspace/",
        'CMD ["sh", "-c", "sleep infinity"]',
    ]
    require(
        _docker_instructions(dockerfile) == expected,
        f"{task_name}: Dockerfile operational contract drift",
    )


def _require_task_toml_contract(task_name: str, task_toml: dict) -> None:
    expected = {
        "version": "1.0",
        "metadata": {
            "author_name": "MLS-Bench",
            "difficulty": "hard",
            "category": "ml-research",
        },
        "task": {"name": task_name},
        "agent": {"timeout_sec": 1800},
        "verifier": {"timeout_sec": 16320},
        "environment": {
            "allow_internet": False,
            "build_timeout_sec": 3600,
            "cpus": 8,
            "memory_mb": 131072,
            "storage_mb": 81920,
            "gpus": 1,
            "gpu_types": ["H20"],
        },
    }
    require(task_toml == expected, f"{task_name}: task.toml contract drift")


def _require_canonical_file(
    task_name: str,
    rendered_path: Path,
    canonical_path: Path,
    label: str,
) -> None:
    require(canonical_path.is_file(), f"missing canonical {label}: {canonical_path}")
    require(rendered_path.is_file(), f"{task_name}: missing rendered {label}")
    require(
        digest(rendered_path) == digest(canonical_path),
        f"{task_name}: rendered {label} drift",
    )


def _copytree_inventory(root: Path) -> list[str]:
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.relative_to(root).parts
        and path.suffix != ".pyc"
    )


def _require_canonical_tree(
    task_name: str,
    rendered_root: Path,
    canonical_root: Path,
    label: str,
) -> list[str]:
    require(canonical_root.is_dir(), f"missing canonical {label}: {canonical_root}")
    require(rendered_root.is_dir(), f"{task_name}: missing rendered {label}")
    canonical_inventory = _copytree_inventory(canonical_root)
    rendered_inventory = sorted(
        path.relative_to(rendered_root).as_posix()
        for path in rendered_root.rglob("*")
        if path.is_file()
    )
    require(
        rendered_inventory == canonical_inventory,
        f"{task_name}: rendered {label} inventory drift",
    )
    for relative in canonical_inventory:
        require(
            digest(rendered_root / relative) == digest(canonical_root / relative),
            f"{task_name}: rendered {label} drift for {relative}",
        )
        rendered_mode = (rendered_root / relative).stat().st_mode & 0o777
        canonical_mode = (canonical_root / relative).stat().st_mode & 0o777
        require(
            rendered_mode == canonical_mode,
            f"{task_name}: rendered {label} mode drift for {relative}",
        )
    return canonical_inventory


def _require_exact_bytes(
    task_name: str,
    path: Path,
    expected: bytes,
    label: str,
) -> None:
    require(path.is_file(), f"{task_name}: missing {label}")
    require(path.read_bytes() == expected, f"{task_name}: {label} drift")


def _require_exact_inventory(
    task_name: str,
    root: Path,
    expected: set[str],
    label: str,
) -> None:
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    require(actual == expected, f"{task_name}: {label} inventory drift")


def _require_mode(
    task_name: str,
    path: Path,
    expected: int | set[int],
    label: str,
) -> None:
    require(path.is_file(), f"{task_name}: missing {label}")
    actual = path.stat().st_mode & 0o777
    expected_modes = {expected} if isinstance(expected, int) else expected
    require(
        actual in expected_modes,
        f"{task_name}: {label} mode is {actual:o}, expected one of "
        f"{sorted(f'{mode:o}' for mode in expected_modes)}",
    )


def _require_regular_tree(task_name: str, root: Path, label: str) -> None:
    require(root.is_dir(), f"{task_name}: missing {label}")
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        relative = path.relative_to(root).as_posix()
        require(
            stat.S_ISREG(mode) or stat.S_ISDIR(mode),
            f"{task_name}: {label}/{relative} is not a regular file or directory",
        )


def _manual_task_digest(task_dir: Path) -> str:
    files = []
    for relative in ("task.toml", "instruction.md", "README.md"):
        path = task_dir / relative
        if path.exists():
            files.append(path)
    for relative in ("environment", "tests", "solution", "steps"):
        root = task_dir / relative
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            task_relative = path.relative_to(task_dir)
            if "__pycache__" in task_relative.parts:
                continue
            if path.suffix in {".pyc", ".pyo"}:
                continue
            files.append(path)
    digest_value = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(task_dir).as_posix()):
        relative = path.relative_to(task_dir).as_posix()
        file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        digest_value.update(f"{relative}\0{file_digest}\n".encode())
    return digest_value.hexdigest()


def _require_dataset_manifest(render_root: Path) -> None:
    manifest_path = render_root / "dataset.toml"
    require(manifest_path.is_file(), "missing rendered dataset.toml")
    _require_mode("render", manifest_path, {0o644, 0o664}, "dataset.toml")
    manifest = tomllib.loads(manifest_path.read_text())
    require(set(manifest) == {"dataset", "tasks"}, "dataset.toml top-level drift")
    dataset = manifest["dataset"]
    require(
        dataset.get("name") == "mls-bench/mls-bench",
        "dataset.toml dataset name drift",
    )
    description = str(dataset.get("description", ""))
    require(
        "10" in description and "140" not in description,
        "dataset.toml description does not describe the 10-task subset",
    )
    require(
        dataset.get("authors")
        == [{"name": "MLS-Bench authors", "email": "bohan22@stanford.edu"}],
        "dataset.toml authors drift",
    )
    require(
        dataset.get("keywords")
        == ["ml-research", "algorithm-design", "multi-seed"],
        "dataset.toml keywords drift",
    )
    tasks = manifest["tasks"]
    require(isinstance(tasks, list), "dataset.toml tasks must be a list")
    require(
        [entry.get("name") for entry in tasks] == list(TASKS),
        "dataset.toml task inventory or order drift",
    )
    for entry in tasks:
        task_name = entry["name"]
        require(
            set(entry) == {"name", "digest"},
            f"dataset.toml entry drift for {task_name}",
        )
        expected = f"sha256:{_manual_task_digest(render_root / task_name)}"
        require(
            entry["digest"] == expected,
            f"dataset.toml digest drift for {task_name}",
        )

    rendering = render_root / ".rendering"
    require(rendering.is_dir(), "missing .rendering directory")
    require(not any(rendering.iterdir()), ".rendering is not empty after render")
    require(
        {path.name for path in render_root.iterdir()}
        == {"dataset.toml", ".rendering", *TASKS},
        "render root inventory drift",
    )


def _require_pristine_contract(
    task_name: str,
    meta: Path,
    scaffold_files: dict[str, bytes],
    declared_files: set[str],
) -> None:
    require(declared_files <= set(scaffold_files), f"{task_name}: undeclared scaffold state")
    expected_manifest = {
        relative: hashlib.sha256(content).hexdigest()
        for relative, content in scaffold_files.items()
    }
    expected_manifest_bytes = (
        json.dumps(expected_manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    _require_exact_bytes(
        task_name,
        meta / "pristine_manifest.json",
        expected_manifest_bytes,
        "pristine_manifest.json",
    )
    pristine_root = meta / "pristine"
    _require_exact_inventory(
        task_name,
        pristine_root,
        declared_files,
        "pristine",
    )
    for relative in declared_files:
        _require_exact_bytes(
            task_name,
            pristine_root / relative,
            scaffold_files[relative],
            f"pristine/{relative}",
        )


def _require_solution_contract(
    task_name: str,
    task: Path,
    source: Path,
    source_root: Path,
    source_config: dict,
) -> None:
    baseline_name = NATIVE_BASELINES[task_name]
    baseline = source_config["baselines"][baseline_name]
    ops_path = source / baseline["edit_ops"]
    ops_namespace = runpy.run_path(str(ops_path))
    require(isinstance(ops_namespace.get("OPS"), list), f"{task_name}: invalid source OPS")
    expected_ops = json.dumps(ops_namespace["OPS"], indent=2).encode()
    solution_root = task / "solution"
    _require_exact_inventory(
        task_name,
        solution_root,
        {"baseline_edit_ops.json", "solve.sh"},
        "solution",
    )
    _require_exact_bytes(
        task_name,
        solution_root / "baseline_edit_ops.json",
        expected_ops,
        "baseline_edit_ops.json",
    )
    _require_mode(
        task_name,
        solution_root / "baseline_edit_ops.json",
        {0o644, 0o664},
        "baseline_edit_ops.json",
    )

    template = (
        source_root
        / "harbor_adapter"
        / "src"
        / "mls_bench"
        / "task-template"
        / "solution"
        / "solve.sh.j2"
    ).read_text()
    expected_solve = (
        template.replace("{{ task_id }}", task_name)
        .replace("{{ baseline_name }}", baseline_name)
        .replace("{{ workdir }}", "/workspace")
    )
    require(
        "{{" not in expected_solve and "{%" not in expected_solve,
        "solve.sh template has unsupported dynamic fields",
    )
    _require_exact_bytes(
        task_name,
        solution_root / "solve.sh",
        expected_solve.encode(),
        "solve.sh",
    )
    _require_mode(
        task_name, solution_root / "solve.sh", {0o755, 0o775}, "solve.sh"
    )


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: audit_summ_rendered.py RENDER_ROOT SOURCE_ROOT")
    render_root = Path(sys.argv[1]).resolve()
    source_root = Path(sys.argv[2]).resolve()
    require(render_root.is_dir(), f"missing render root: {render_root}")
    _require_regular_tree("render", render_root, "render root")

    rendered_tasks = sorted(path.name for path in render_root.glob("summ-*") if path.is_dir())
    require(rendered_tasks == list(TASKS), f"wrong rendered inventory: {rendered_tasks}")
    source_configs = {
        task_name: json.loads(
            (source_root / "tasks" / task_name / "config.json").read_text()
        )
        for task_name in TASKS
    }
    all_editable_files = {
        config["files"][0]["filename"] for config in source_configs.values()
    }
    canonical_tests = (
        source_root / "harbor_adapter" / "src" / "mls_bench" / "task-template" / "tests"
    )
    canonical_score_task = canonical_tests / "score_task.py"
    canonical_test_runner = canonical_tests / "test.sh"
    canonical_mlsbench = source_root / "src" / "mlsbench"
    canonical_mlsbench_inventory = _copytree_inventory(canonical_mlsbench)
    require(canonical_mlsbench_inventory, "canonical MLS-Bench source tree is empty")
    package_config = json.loads(
        (
            source_root
            / "vendor"
            / "pkg_configs"
            / PACKAGE
            / "config.json"
        ).read_text()
    )
    expected_package_envs = json.dumps(
        {PACKAGE: package_config.get("env") or {}}, indent=2
    ).encode()
    runtime_total = 0
    scaffold_total = 0
    mlsbench_source_total = 0

    for task_name in TASKS:
        task = render_root / task_name
        source = source_root / "tasks" / task_name
        source_config = source_configs[task_name]
        editable = Path(source_config["files"][0]["filename"])
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

        _require_canonical_file(
            task_name,
            task / "tests" / "score_task.py",
            canonical_score_task,
            "score_task.py",
        )
        _require_mode(
            task_name,
            task / "tests" / "score_task.py",
            canonical_score_task.stat().st_mode & 0o777,
            "score_task.py",
        )
        _require_canonical_file(
            task_name,
            task / "tests" / "test.sh",
            canonical_test_runner,
            "test.sh",
        )
        _require_mode(
            task_name,
            task / "tests" / "test.sh",
            (canonical_test_runner.stat().st_mode & 0o777) | 0o111,
            "test.sh",
        )
        rendered_mlsbench_inventory = _require_canonical_tree(
            task_name,
            task / "tests" / "mlsbench_src" / "mlsbench",
            canonical_mlsbench,
            "mlsbench_src",
        )
        require(
            rendered_mlsbench_inventory == canonical_mlsbench_inventory,
            f"{task_name}: inconsistent canonical MLS-Bench inventory",
        )
        mlsbench_source_total += len(rendered_mlsbench_inventory)

        dockerfile = (task / "environment" / "Dockerfile").read_text()
        _require_dockerfile_contract(task_name, dockerfile)
        _require_mode(
            task_name,
            task / "environment" / "Dockerfile",
            {0o644, 0o664},
            "Dockerfile",
        )

        task_toml = tomllib.loads((task / "task.toml").read_text())
        _require_task_toml_contract(task_name, task_toml)
        _require_mode(
            task_name, task / "task.toml", {0o644, 0o664}, "task.toml"
        )

        instruction_text = (task / "instruction.md").read_text()
        _require_mode(
            task_name,
            task / "instruction.md",
            {0o644, 0o664},
            "instruction.md",
        )
        instruction = instruction_text.lower()
        require(
            (source / "task_description.md").read_text() in instruction_text,
            f"{task_name}: source task description is not embedded verbatim",
        )
        for setting in ("xsum", "cnn/dailymail", "samsum"):
            require(setting in instruction, f"{task_name}: instruction omits {setting}")
        require("512" in instruction, f"{task_name}: instruction omits token cap")
        require("hidden setting" not in instruction, f"{task_name}: hidden-setting semantics leaked")
        require("public setting" not in instruction, f"{task_name}: public-setting semantics leaked")

        config = json.loads((task / "tests" / "meta" / "config.json").read_text())
        _require_source_config_contract(
            task_name, config, source_config, all_editable_files
        )
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
        require(
            digest(task / "tests" / "meta" / "leaderboard.csv")
            == digest(source / "leaderboard.csv"),
            f"{task_name}: rendered leaderboard drift",
        )
        _require_mode(
            task_name,
            task / "tests" / "meta" / "config.json",
            {0o644, 0o664},
            "config.json",
        )
        for relative in ("parser.py", "score_spec.py", "leaderboard.csv"):
            _require_mode(
                task_name,
                task / "tests" / "meta" / relative,
                (source / relative).stat().st_mode & 0o777,
                relative,
            )
        source_edits_root = source / "edits"
        rendered_edits_root = task / "tests" / "meta" / "edits"
        source_edits = _require_canonical_tree(
            task_name,
            rendered_edits_root,
            source_edits_root,
            "meta/edits",
        )
        runtime_root = task / "tests" / "meta" / "verifier_package_files"
        expected_runtime = sorted(source_config["verifier_only_package_files"])
        require(
            len(expected_runtime) == 11 and set(expected_runtime) == VERIFIER_RUNTIME,
            f"{task_name}: source verifier-only runtime contract drift",
        )
        rendered_runtime_inventory = sorted(
            path.relative_to(runtime_root).as_posix()
            for path in runtime_root.rglob("*")
            if path.is_file()
        )
        require(
            rendered_runtime_inventory == expected_runtime,
            f"{task_name}: verifier-only runtime inventory drift",
        )
        runtime_total += len(expected_runtime)
        for relative in expected_runtime:
            rendered_runtime = runtime_root / relative
            source_runtime = source_root / "vendor" / relative
            require(
                rendered_runtime.is_file(),
                f"{task_name}: missing verifier-only runtime {relative}",
            )
            require(
                digest(rendered_runtime) == digest(source_runtime),
                f"{task_name}: verifier-only runtime drift for {relative}",
            )
            rendered_mode = rendered_runtime.stat().st_mode & 0o777
            source_mode = source_runtime.stat().st_mode & 0o777
            require(
                rendered_mode == source_mode,
                f"{task_name}: verifier-only runtime mode drift for {relative}",
            )

        meta = task / "tests" / "meta"
        fixed_meta = {
            "task_id": f"{task_name}\n".encode(),
            "package": f"{PACKAGE}\n".encode(),
            "workdir": b"/workspace\n",
            "gpu_count": b"1\n",
            "package_envs.json": expected_package_envs,
        }
        for relative, expected_bytes in fixed_meta.items():
            _require_exact_bytes(
                task_name,
                meta / relative,
                expected_bytes,
                f"meta/{relative}",
            )
            _require_mode(
                task_name,
                meta / relative,
                {0o644, 0o664},
                f"meta/{relative}",
            )
        require(
            not (meta / "gpu_compute_cap").exists(),
            f"{task_name}: unexpected GPU compute cap",
        )
        require(
            not (meta / "budget_check.py").exists(),
            f"{task_name}: unexpected budget check",
        )

        expected_verifier_manifest = (
            json.dumps(
                {
                    "version": 1,
                    "entries": [
                        {"path": relative, "kind": "file"}
                        for relative in expected_runtime
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode()
        _require_exact_bytes(
            task_name,
            meta / "verifier_package_manifest.json",
            expected_verifier_manifest,
            "meta/verifier_package_manifest.json",
        )
        _require_mode(
            task_name,
            meta / "verifier_package_manifest.json",
            {0o644, 0o664},
            "meta/verifier_package_manifest.json",
        )

        source_scripts_root = source / "scripts"
        rendered_meta_scripts = _require_canonical_tree(
            task_name,
            meta / "scripts",
            source_scripts_root,
            "meta/scripts",
        )
        score_text = (task / "tests" / "meta" / "score_spec.py").read_text()
        for setting in ("xsum", "cnndm", "samsum"):
            require(setting in score_text, f"{task_name}: score omits {setting}")

        eval_script = task / "tests" / "eval" / "scripts" / "run.sh"
        require(eval_script.is_file(), f"{task_name}: missing active eval script")
        eval_root = task / "tests" / "eval"
        require(
            sorted(
                path.relative_to(eval_root).as_posix()
                for path in eval_root.rglob("*")
                if path.is_file()
            )
            == ["scripts/run.sh"],
            f"{task_name}: active eval inventory drift",
        )
        require(
            digest(eval_script) == digest(source / "scripts" / "run.sh"),
            f"{task_name}: active eval script drift",
        )
        source_script_mode = (source / "scripts" / "run.sh").stat().st_mode & 0o777
        _require_mode(task_name, eval_script, source_script_mode, "active eval script")
        _require_mode(
            task_name,
            meta / "scripts" / "run.sh",
            source_script_mode,
            "meta eval script",
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
        scaffold_root = task / "environment" / "_scaffold"
        require(
            sorted(
                path.relative_to(scaffold_root).as_posix()
                for path in scaffold_root.rglob("*")
                if path.is_file()
            )
            == sorted([
                "abstractive-summarization/__init__.py",
                editable.as_posix(),
            ]),
            f"{task_name}: agent scaffold inventory drift",
        )
        scaffold_total += 1
        require(
            (
                scaffold_root
                / "abstractive-summarization"
                / "__init__.py"
            ).read_bytes()
            == b"\n",
            f"{task_name}: scaffold package initializer is not canonical",
        )
        _require_mode(
            task_name,
            scaffold_root / PACKAGE / "__init__.py",
            {0o644, 0o664},
            "scaffold package initializer",
        )
        require(
            digest(scaffold_root / editable)
            == digest(source / "edits" / "custom_template.py"),
            f"{task_name}: rendered editable scaffold drift",
        )
        _require_mode(
            task_name,
            scaffold_root / editable,
            {0o644, 0o664},
            "editable scaffold",
        )
        scaffold_files = {
            f"{PACKAGE}/__init__.py": b"\n",
            editable.as_posix(): (source / "edits" / "custom_template.py").read_bytes(),
        }
        _require_pristine_contract(
            task_name,
            meta,
            scaffold_files,
            {editable.as_posix()},
        )
        _require_mode(
            task_name,
            meta / "pristine_manifest.json",
            {0o644, 0o664},
            "pristine_manifest.json",
        )
        _require_mode(
            task_name,
            meta / "pristine" / editable,
            {0o644, 0o664},
            "pristine editable",
        )
        _require_solution_contract(
            task_name,
            task,
            source,
            source_root,
            source_config,
        )

        environment_inventory = {
            "Dockerfile",
            f"_scaffold/{PACKAGE}/__init__.py",
            f"_scaffold/{editable.as_posix()}",
        }
        _require_exact_inventory(
            task_name,
            task / "environment",
            environment_inventory,
            "environment",
        )

        meta_inventory = {
            "config.json",
            "gpu_count",
            "leaderboard.csv",
            "package",
            "package_envs.json",
            "parser.py",
            f"pristine/{editable.as_posix()}",
            "pristine_manifest.json",
            "score_spec.py",
            "task_id",
            "verifier_package_manifest.json",
            "workdir",
        }
        meta_inventory.update(f"edits/{relative}" for relative in source_edits)
        meta_inventory.update(
            f"scripts/{relative}" for relative in rendered_meta_scripts
        )
        meta_inventory.update(
            f"verifier_package_files/{relative}" for relative in expected_runtime
        )
        expected_tests_inventory = {
            "score_task.py",
            "test.sh",
            *{f"eval/scripts/{relative}" for relative in rendered_meta_scripts},
            *{f"meta/{relative}" for relative in meta_inventory},
            *{
                f"mlsbench_src/mlsbench/{relative}"
                for relative in canonical_mlsbench_inventory
            },
        }
        _require_exact_inventory(
            task_name,
            task / "tests",
            expected_tests_inventory,
            "tests",
        )
        expected_task_inventory = {
            "instruction.md",
            "task.toml",
            *{f"environment/{relative}" for relative in environment_inventory},
            "solution/baseline_edit_ops.json",
            "solution/solve.sh",
            *{f"tests/{relative}" for relative in expected_tests_inventory},
        }
        _require_exact_inventory(
            task_name,
            task,
            expected_task_inventory,
            "task artifact",
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

    _require_dataset_manifest(render_root)
    require(runtime_total == 110, f"wrong verifier runtime total: {runtime_total}")
    require(scaffold_total == 10, f"wrong scaffold total: {scaffold_total}")
    require(
        mlsbench_source_total == 10 * len(canonical_mlsbench_inventory),
        f"wrong rendered MLS-Bench source total: {mlsbench_source_total}",
    )
    print(
        "SUMM_RENDER_AUDIT tasks=10 parsers=10 score_specs=10 scripts=10 "
        f"verifier_runtime={runtime_total} scaffolds={scaffold_total} "
        f"mlsbench_src={mlsbench_source_total} "
        "gpu=1 h20=10 settings=30 "
        "no_leak=10 image_pinned=10"
    )


if __name__ == "__main__":
    main()
