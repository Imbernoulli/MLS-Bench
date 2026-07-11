from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
ADAPTER_SRC = ROOT / "harbor_adapter" / "src"
if str(ADAPTER_SRC) not in sys.path:
    sys.path.insert(0, str(ADAPTER_SRC))

from mls_bench.adapter import (  # noqa: E402
    MlsBenchAdapter,
    MlsBenchRoot,
    _agent_data_prune_paths,
    _agent_image_prune_paths,
    _base_image,
    _mangrove_resources,
    build_task_context,
    render_task,
)


PINNED_IMAGE = "registry.example/mlsbench-pkg@sha256:" + "a" * 64


def _write_privacy_repo(root: Path) -> MlsBenchRoot:
    task = root / "tasks" / "privacy-task"
    edits = task / "edits"
    scripts = task / "scripts"
    package = root / "vendor" / "pkg"
    package_config = root / "vendor" / "pkg_configs" / "pkg"
    data_root = root / "private-data"
    for directory in (edits, scripts, package / "solution", package / "baselines", package_config, data_root):
        directory.mkdir(parents=True, exist_ok=True)

    (root / "vendor" / "packages.yaml").write_text("{}\n")
    (package / "__init__.py").write_text("")
    (package / "util.py").write_text("VALUE = 1\n")
    (package / "harness.py").write_text("VALUE = 'source'\n")
    (package / "legacy.py").write_text("LEGACY = True\n")
    (package / "manual-secret.py").write_text("SECRET = True\n")
    (package / "baselines" / "answer.py").write_text("ANSWER = 42\n")
    (package / "solution" / "current.py").write_text("CHOICE = 'native'\n")
    (package / "solution" / "sibling.py").write_text("CHOICE = 'sibling-answer'\n")
    (data_root / "private.bin").write_bytes(b"private-evaluation-data")

    package_config.write_text(
        json.dumps(
            {
                "mangrove_base_image": PINNED_IMAGE,
                "use_cuda": True,
                "workdir": "/workspace",
                "agent_pruned_files": ["manual-secret.py"],
            }
        )
    )
    config = {
        "agent_image_prune": ["/opt/private-proof"],
        "agent_data_prune": ["/data/pkg/private.bin"],
        "agent_pruned_package_files": ["pkg/legacy.py"],
        "verifier_only_package_files": ["pkg/harness.py"],
        "verifier_data_deps": [
            {
                "name": "private-data",
                "host_path": str(data_root / "private.bin"),
                "dest": "data/pkg/private.bin",
                "required": True,
            }
        ],
        "test_cmds": [
            {
                "cmd": "scripts/run.sh",
                "label": "required-setting",
                "group": 1,
                "compute": 1,
                "time": "4:00:00",
                "mem": 48,
                "package": "pkg",
                "hidden": True,
            }
        ],
        "seeds": [42],
        "files": [
            {
                "filename": "pkg/solution/current.py",
                "read": [{"start": -1, "end": -1}],
                "edit": [{"start": -1, "end": -1}],
            }
        ],
    }
    (task / "config.json").write_text(json.dumps(config))
    (task / "task_description.md").write_text("Implement the declared research surface.\n")
    (task / "parser.py").write_text("class Parser:\n    pass\n")
    (task / "score_spec.py").write_text("SCORE_SPEC = None\n")
    (task / "leaderboard.csv").write_text("model\n")
    (scripts / "run.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
    (edits / "mid_edit.py").write_text(
        "OPS = [{\"op\": \"replace\", \"file\": \"pkg/harness.py\", "
        "\"start_line\": 1, \"end_line\": 1, "
        "\"content\": \"VALUE = 'task'\\n\"}]\n"
    )
    return MlsBenchRoot(root)


def test_mangrove_render_is_pinned_private_h20_and_offline(tmp_path: Path):
    mb = _write_privacy_repo(tmp_path / "repo")
    ctx = build_task_context(mb, "privacy-task")
    rendered = render_task(
        mb,
        ctx,
        tmp_path / "rendered",
        mangrove=True,
        gpu_backend="h20",
    )

    dockerfile = (rendered / "environment" / "Dockerfile").read_text()
    assert f"FROM {PINNED_IMAGE}" in dockerfile
    assert "RUN rm -rf /opt/private-proof" in dockerfile
    assert "RUN rm -rf /data/pkg/private.bin" in dockerfile
    assert "RUN rm -rf /workspace/pkg" in dockerfile
    assert "COPY _scaffold/ /workspace/" in dockerfile
    assert "pip install" not in dockerfile

    scaffold = rendered / "environment" / "_scaffold" / "pkg"
    assert (scaffold / "util.py").is_file()
    assert (scaffold / "solution" / "current.py").is_file()
    assert not (scaffold / "harness.py").exists()
    assert not (scaffold / "legacy.py").exists()
    assert not (scaffold / "manual-secret.py").exists()
    assert not (scaffold / "baselines").exists()
    assert not (scaffold / "solution" / "sibling.py").exists()

    meta = rendered / "tests" / "meta"
    assert (meta / "verifier_package_files" / "pkg" / "harness.py").read_text() == "VALUE = 'task'\n"
    assert (meta / "data" / "pkg" / "private.bin").read_bytes() == b"private-evaluation-data"
    manifest = json.loads((meta / "pristine_manifest.json").read_text())
    assert "pkg/util.py" in manifest
    assert "pkg/solution/current.py" in manifest
    assert "pkg/harness.py" not in manifest
    assert "pkg/legacy.py" not in manifest
    assert "pkg/solution/sibling.py" not in manifest

    rendered_config = json.loads((meta / "config.json").read_text())
    assert "hidden" not in rendered_config["test_cmds"][0]
    task_toml = tomllib.loads((rendered / "task.toml").read_text())
    environment = task_toml["environment"]
    assert environment["allow_internet"] is False
    assert environment["gpus"] == 1
    assert environment["gpu_types"] == ["H20"]
    assert environment["cpus"] == 8
    assert environment["memory_mb"] == 131072
    assert not (rendered / "environment" / "docker-compose.yaml").exists()
    instruction = (rendered / "instruction.md").read_text().lower()
    assert "required-setting" not in instruction
    assert "public" not in instruction
    assert "hidden" not in instruction


def test_mangrove_image_must_be_one_immutable_digest():
    assert _base_image("pkg", True, {"pinned_harbor_image": PINNED_IMAGE}) == PINNED_IMAGE
    assert _base_image(
        "pkg",
        True,
        {"pinned_harbor_image": PINNED_IMAGE, "mangrove_base_image": PINNED_IMAGE},
    ) == PINNED_IMAGE

    with pytest.raises(ValueError, match="refusing the mutable :latest fallback"):
        _base_image("pkg", True, {})
    with pytest.raises(ValueError, match="must be pinned"):
        _base_image("pkg", True, {"mangrove_base_image": "registry.example/pkg:latest"})
    with pytest.raises(ValueError, match="conflicting"):
        _base_image(
            "pkg",
            True,
            {
                "mangrove_base_image": PINNED_IMAGE,
                "pinned_harbor_image": "registry.example/other@sha256:" + "b" * 64,
            },
        )


def test_h20_is_default_serial_is_one_card_and_task_cap_is_four(tmp_path: Path):
    assert MlsBenchAdapter(tmp_path).gpu_backend == "h20"
    package = {"use_cuda": True}
    entries = [
        {
            "cmd": f"scripts/{idx}.sh",
            "label": str(idx),
            "group": 1,
            "compute": 1,
            "time": "0:05:00",
            "package": "pkg",
        }
        for idx in range(5)
    ]
    serial = _mangrove_resources(
        package,
        {"test_cmds": entries, "seeds": [42], "_verifier_serial": True},
    )
    assert serial["gpus"] == 1
    assert serial["gpu_types"] == ["H20"]

    with pytest.raises(ValueError, match="4-GPU task cap"):
        _mangrove_resources(package, {"test_cmds": entries, "seeds": [42]})


def test_evidence_based_resource_overrides_merge_package_then_task():
    package = {
        "use_cuda": True,
        "mangrove_resources": {
            "cpus": 6,
            "memory_mb": 32768,
            "storage_mb": 61440,
        },
    }
    task = {
        "mangrove_resources": {"cpus": 4},
        "test_cmds": [
            {
                "cmd": "scripts/run.sh",
                "label": "setting",
                "group": 1,
                "compute": 1,
                "time": "0:10:00",
                "package": "pkg",
            }
        ],
    }
    resources = _mangrove_resources(package, task)
    assert resources["gpus"] == 1
    assert resources["cpus"] == 4
    assert resources["memory_mb"] == 32768
    assert resources["storage_mb"] == 61440

    with pytest.raises(ValueError, match="unsupported field"):
        _mangrove_resources(
            {"use_cuda": True},
            {**task, "mangrove_resources": {"gpu_types": ["H20"]}},
        )
    with pytest.raises(ValueError, match="positive integer"):
        _mangrove_resources(
            {"use_cuda": True},
            {**task, "mangrove_resources": {"cpus": True}},
        )


@pytest.mark.parametrize(
    ("function", "config", "message"),
    [
        (_agent_data_prune_paths, {"agent_data_prune": ["/data/../etc/passwd"]}, "normal absolute"),
        (_agent_data_prune_paths, {"agent_data_prune": ["/opt/file"]}, "below /data"),
        (_agent_image_prune_paths, {"agent_image_prune": ["/opt"]}, "below /opt"),
        (_agent_image_prune_paths, {"agent_image_prune": "not-a-list"}, "must be a list"),
    ],
)
def test_prune_paths_reject_unsafe_values(function, config, message):
    with pytest.raises(ValueError, match=message):
        function(config)
