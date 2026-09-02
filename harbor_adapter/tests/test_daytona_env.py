from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths


def _module():
    # Keep the test independent of whether the caller installed the adapter as
    # a package or is using the documented PYTHONPATH=src workflow.
    import sys

    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import mls_bench.harbor_env as module

    return module


def _compose(count: int = 2) -> str:
    return yaml.safe_dump(
        {
            "services": {
                "main": {
                    "deploy": {
                        "resources": {
                            "reservations": {
                                "devices": [
                                    {
                                        "driver": "nvidia",
                                        "count": count,
                                        "capabilities": ["gpu"],
                                    }
                                ]
                            }
                        }
                    }
                }
            }
        },
        sort_keys=False,
    )


def test_gpu_reservation_only_compose_is_strict(tmp_path: Path):
    module = _module()
    env = tmp_path / "environment"
    env.mkdir()
    (env / "docker-compose.yaml").write_text(_compose(2))

    assert module.is_gpu_reservation_only_compose(env, gpu_count=2)
    assert not module.is_gpu_reservation_only_compose(env, gpu_count=1)

    (env / "docker-compose.yaml").write_text(
        _compose(2).replace("services:\n", "services:\n  sidecar:\n    image: redis:7\n")
    )
    assert not module.is_gpu_reservation_only_compose(env, gpu_count=2)


def test_gpu_compose_uses_direct_strategy_and_restores_resources(tmp_path: Path):
    module = _module()
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    (env_dir / "docker-compose.yaml").write_text(_compose(2))
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()

    config = EnvironmentConfig(
        cpus=2,
        memory_mb=4096,
        storage_mb=10240,
        gpus=2,
    )
    env = module.DaytonaEnvironment(
        environment_dir=env_dir,
        environment_name="gpu-task",
        session_id="gpu-task.1",
        trial_paths=trial_paths,
        task_env_config=config,
    )

    assert env.task_env_config.gpus == 2
    assert env._compose_mode is False
    assert env._strategy.__class__.__name__ == "_GpuAwareDirect"
    assert env.capabilities.gpus is True

    captured = []

    async def fake_get_instance():
        return SimpleNamespace(get_client=lambda: asyncio.sleep(0, result=object()))

    async def fake_create_sandbox(*, params):
        captured.append(params)
        env._sandbox = object()

    async def fake_exec(*args, **kwargs):
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(module.DaytonaClientManager, "get_instance", fake_get_instance)
        monkeypatch.setattr(env, "_configure_daytona_client", fake_exec)
        monkeypatch.setattr(env, "_create_sandbox", fake_create_sandbox)
        monkeypatch.setattr(env, "_sandbox_exec", fake_exec)
        asyncio.run(env.start(force_build=True))
    finally:
        monkeypatch.undo()

    assert len(captured) == 1
    assert captured[0].resources.gpu == 2


def test_gpu_override_keeps_reservation_only_compose_on_direct_path(tmp_path: Path):
    module = _module()
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    (env_dir / "docker-compose.yaml").write_text(_compose(2))
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()

    config = EnvironmentConfig(cpus=2, memory_mb=4096, storage_mb=10240, gpus=2)
    env = module.DaytonaEnvironment(
        environment_dir=env_dir,
        environment_name="gpu-override-task",
        session_id="gpu-override-task.1",
        trial_paths=trial_paths,
        task_env_config=config,
        override_gpus=1,
    )

    assert env.task_env_config.gpus == 1
    assert env._compose_mode is False
    assert env._strategy.__class__.__name__ == "_GpuAwareDirect"

    # An explicit zero-GPU override still uses the direct Dockerfile path; it
    # must not fall back to Daytona DinD merely because the overlay's original
    # reservation requested devices.
    config_zero = EnvironmentConfig(cpus=2, memory_mb=4096, storage_mb=10240, gpus=2)
    env_zero = module.DaytonaEnvironment(
        environment_dir=env_dir,
        environment_name="gpu-override-zero-task",
        session_id="gpu-override-zero-task.1",
        trial_paths=trial_paths,
        task_env_config=config_zero,
        override_gpus=0,
    )
    assert env_zero.task_env_config.gpus == 0
    assert env_zero._compose_mode is False
    assert env_zero._strategy.__class__.__name__ == "_GpuAwareDirect"


def test_spot_h200_forwards_type_and_verifier_environment(tmp_path: Path):
    module = _module()
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    (env_dir / "docker-compose.yaml").write_text(_compose(1))
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    config = EnvironmentConfig(
        cpus=2, memory_mb=4096, storage_mb=10240, gpus=1
    )
    env = module.DaytonaEnvironment(
        environment_dir=env_dir,
        environment_name="spot-h200-task",
        session_id="spot-h200-task.1",
        trial_paths=trial_paths,
        task_env_config=config,
        spot=True,
        gpu_type="H200",
    )

    captured = []

    async def fake_get_instance():
        return SimpleNamespace(get_client=lambda: asyncio.sleep(0, result=object()))

    async def fake_parent_create_sandbox(self, params, *args, **kwargs):
        captured.append(params)
        self._sandbox = object()

    async def fake_exec(*args, **kwargs):
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(module.DaytonaClientManager, "get_instance", fake_get_instance)
        monkeypatch.setattr(env, "_configure_daytona_client", fake_exec)
        monkeypatch.setattr(
            module._HarborDaytonaEnvironment,
            "_create_sandbox",
            fake_parent_create_sandbox,
        )
        monkeypatch.setattr(env, "_sandbox_exec", fake_exec)
        # Call the adapter hook directly so the assertions exercise the
        # spot/GPU-type/env-var mutation rather than replacing the hook.
        from daytona import CreateSandboxFromImageParams, Image, Resources

        params = CreateSandboxFromImageParams(
            image=Image.base("ubuntu:22.04"),
            resources=Resources(cpu=2, memory=4, disk=10, gpu=1),
        )
        asyncio.run(env._create_sandbox(params))
    finally:
        monkeypatch.undo()

    assert len(captured) == 1
    assert captured[0].spot is True
    assert captured[0].resources.gpu == 1
    assert captured[0].resources.gpu_type.name == "H200"
    assert captured[0].env_vars["MLSBENCH_GPU_TYPE"] == "H200"


def test_h200_uses_native_override_compute_for_daytona_reservation(tmp_path: Path):
    """H200 capacity comes from the rendered task's existing config metadata."""
    module = _module()
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    (env_dir / "docker-compose.yaml").write_text(_compose(2))
    meta = tmp_path / "tests" / "meta"
    meta.mkdir(parents=True)
    (meta / "config.json").write_text(
        json.dumps(
            {
                "test_cmds": [
                    {
                        "cmd": "scripts/train.sh",
                        "compute": 2,
                        "group": 1,
                        "h200": {
                            "compute": 1,
                            "env": {"TP_SIZE": "1"},
                        },
                    }
                ],
                "seeds": [42],
            }
        )
    )
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    config = EnvironmentConfig(cpus=2, memory_mb=4096, storage_mb=10240, gpus=2)
    env = module.DaytonaEnvironment(
        environment_dir=env_dir,
        environment_name="native-h200-resource-task",
        session_id="native-h200-resource-task.1",
        trial_paths=trial_paths,
        task_env_config=config,
        spot=True,
        gpu_type="H200",
    )

    # The verifier still reads the same h200 block; only Daytona's resource
    # request is reduced from the H100 baseline of two cards to one.
    assert env.task_env_config.gpus == 1


def test_h100_keeps_native_baseline_reservation_with_h200_metadata(tmp_path: Path):
    module = _module()
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    (env_dir / "docker-compose.yaml").write_text(_compose(2))
    meta = tmp_path / "tests" / "meta"
    meta.mkdir(parents=True)
    (meta / "config.json").write_text(
        json.dumps(
            {
                "test_cmds": [
                    {
                        "cmd": "scripts/train.sh",
                        "compute": 2,
                        "h200": {"compute": 1},
                    }
                ]
            }
        )
    )
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    config = EnvironmentConfig(cpus=2, memory_mb=4096, storage_mb=10240, gpus=2)
    env = module.DaytonaEnvironment(
        environment_dir=env_dir,
        environment_name="native-h100-resource-task",
        session_id="native-h100-resource-task.1",
        trial_paths=trial_paths,
        task_env_config=config,
        spot=True,
        gpu_type="H100",
    )
    assert env.task_env_config.gpus == 2


def test_spot_defaults_to_h100_without_explicit_gpu_type(tmp_path: Path):
    module = _module()
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    (env_dir / "docker-compose.yaml").write_text(_compose(1))
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    config = EnvironmentConfig(cpus=2, memory_mb=4096, storage_mb=10240, gpus=1)
    env = module.DaytonaEnvironment(
        environment_dir=env_dir,
        environment_name="spot-default-task",
        session_id="spot-default-task.1",
        trial_paths=trial_paths,
        task_env_config=config,
        spot=True,
    )

    assert env._gpu_type_requested().name == "H100"


def test_spot_rejects_genuine_gpu_compose(tmp_path: Path):
    module = _module()
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    (env_dir / "docker-compose.yaml").write_text(
        yaml.safe_dump(
            {
                "services": {
                    "main": {"deploy": {"resources": {"reservations": {"devices": [{"driver": "nvidia", "count": 1, "capabilities": ["gpu"]}]}}}},
                    "sidecar": {"image": "redis:7"},
                }
            }
        )
    )
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    config = EnvironmentConfig(cpus=2, memory_mb=4096, storage_mb=10240, gpus=1)
    with pytest.raises(ValueError, match="genuine multi-container Compose"):
        module.DaytonaEnvironment(
            environment_dir=env_dir,
            environment_name="genuine-compose-spot",
            session_id="genuine-compose-spot.1",
            trial_paths=trial_paths,
            task_env_config=config,
            spot=True,
        )


def test_daytona_run_configs_select_custom_environment():
    for path, expected_dataset in (
        (Path("harbor/run-daytona.yaml"), "tasks"),
        (Path("harbor_adapter/run-daytona.yaml"), "datasets/mls-bench"),
    ):
        data = yaml.safe_load(path.read_text())
        assert data["environment"]["import_path"].endswith(
            "DaytonaEnvironment"
        )
        assert data["datasets"][0]["path"] == expected_dataset
        assert set(data["datasets"][0]["exclude_task_names"]) == {
            "mls-bench__agent-tool-reasoning",
            "mls-bench__mas-topology",
        }
