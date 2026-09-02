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


def test_h200_does_not_scale_tasks_without_native_profile(tmp_path: Path):
    module = _module()
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    (env_dir / "docker-compose.yaml").write_text(_compose(2))
    meta = tmp_path / "tests" / "meta"
    meta.mkdir(parents=True)
    (meta / "config.json").write_text(
        json.dumps({"test_cmds": [{"cmd": "scripts/train.sh", "compute": 2}]})
    )
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    config = EnvironmentConfig(cpus=2, memory_mb=4096, storage_mb=10240, gpus=2)
    env = module.DaytonaEnvironment(
        environment_dir=env_dir,
        environment_name="unprofiled-h200-task",
        session_id="unprofiled-h200-task.1",
        trial_paths=trial_paths,
        task_env_config=config,
        spot=True,
        gpu_type="H200",
    )
    assert env.task_env_config.gpus == 2


def test_sandbox_environment_and_resource_floor(tmp_path: Path):
    """Thread caps, NCCL fallback, RAM/CPU floors and time scale reach Daytona."""
    module = _module()
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    (env_dir / "docker-compose.yaml").write_text(_compose(2))
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    config = EnvironmentConfig(cpus=4, memory_mb=16384, storage_mb=61440, gpus=2)
    env = module.DaytonaEnvironment(
        environment_dir=env_dir,
        environment_name="floor-task",
        session_id="floor-task.1",
        trial_paths=trial_paths,
        task_env_config=config,
        gpu_memory_gb="64",
        gpu_cpus="16",
    )
    # No explicit gpu_type and no spot: still Hopper, never Daytona's default.
    assert env._gpu_type_requested().name == "H100"

    captured = []

    async def fake_parent_create_sandbox(self, params, *args, **kwargs):
        captured.append(params)
        self._sandbox = object()

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(
            module._HarborDaytonaEnvironment,
            "_create_sandbox",
            fake_parent_create_sandbox,
        )
        from daytona import CreateSandboxFromImageParams, Image, Resources

        params = CreateSandboxFromImageParams(
            image=Image.base("ubuntu:22.04"),
            resources=Resources(cpu=4, memory=16, disk=60, gpu=2),
        )
        asyncio.run(env._create_sandbox(params))
    finally:
        monkeypatch.undo()

    resources = captured[0].resources
    assert (resources.cpu, resources.memory, resources.disk, resources.gpu) == (16, 64, 60, 2)
    assert resources.gpu_type.name == "H100"
    env_vars = captured[0].env_vars
    assert env_vars["MLSBENCH_GPU_TYPE"] == "H100"
    assert env_vars["OMP_NUM_THREADS"] == "16"
    assert env_vars["MKL_NUM_THREADS"] == "16"
    assert env_vars["NCCL_CUMEM_ENABLE"] == "0"
    assert captured[0].spot is not True


def test_snapshot_salt_appends_noop_layer_and_decoys_default_off(tmp_path: Path):
    """``snapshot_salt`` forces a fresh Daytona snapshot; decoys stay opt-in."""
    module = _module()
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    (env_dir / "docker-compose.yaml").write_text(_compose(1))
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    config = EnvironmentConfig(cpus=4, memory_mb=16384, storage_mb=61440, gpus=1)
    env = module.DaytonaEnvironment(
        environment_dir=env_dir,
        environment_name="salt-task",
        session_id="salt-task.1",
        trial_paths=trial_paths,
        task_env_config=config,
        snapshot_salt="rebuild-1",
    )
    captured = []

    async def fake_parent_create_sandbox(self, params, *args, **kwargs):
        captured.append(params)
        self._sandbox = object()

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(
            module._HarborDaytonaEnvironment,
            "_create_sandbox",
            fake_parent_create_sandbox,
        )
        from daytona import CreateSandboxFromImageParams, Image, Resources

        params = CreateSandboxFromImageParams(
            image=Image.base("ubuntu:22.04"),
            resources=Resources(cpu=4, memory=16, disk=60, gpu=1),
        )
        asyncio.run(env._create_sandbox(params))
    finally:
        monkeypatch.undo()

    assert "mlsbench snapshot salt rebuild-1" in captured[0].image.dockerfile()
    assert env._kwargs.get("toolbox_hold_bad_placements", False) is False


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


def test_verl_bundles_get_the_128g_memory_floor_automatically(tmp_path: Path):
    """A rendered verl task is raised to 128 GiB even when the kwarg says 64.

    verl's validation generation OOM-killed the ray worker in a 64 GiB sandbox
    and completed only at 128 GiB; the floor is keyed on the ``task.toml``
    package metadata so no flag is needed.  Other packages keep the kwarg.
    """
    module = _module()
    from daytona import CreateSandboxFromImageParams, Image, Resources

    for package, expected_memory in (("verl", 128), ("nanoGPT", 64), (None, 64)):
        task_dir = tmp_path / f"task-{package}"
        env_dir = task_dir / "environment"
        env_dir.mkdir(parents=True)
        (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")
        (env_dir / "docker-compose.yaml").write_text(_compose(2))
        if package is not None:
            (task_dir / "task.toml").write_text(
                f'[metadata]\nmls_bench_package = "{package}"\n'
            )
        trial_paths = TrialPaths(task_dir / "trial")
        trial_paths.mkdir()
        env = module.DaytonaEnvironment(
            environment_dir=env_dir,
            environment_name=f"floor-{package}",
            session_id=f"floor-{package}.1",
            trial_paths=trial_paths,
            task_env_config=EnvironmentConfig(
                cpus=4, memory_mb=16384, storage_mb=61440, gpus=2
            ),
            gpu_memory_gb="64",
            gpu_cpus="16",
        )
        captured = []

        async def fake_parent_create_sandbox(self, params, *args, **kwargs):
            captured.append(params)
            self._sandbox = object()

        monkeypatch = pytest.MonkeyPatch()
        try:
            monkeypatch.setattr(
                module._HarborDaytonaEnvironment,
                "_create_sandbox",
                fake_parent_create_sandbox,
            )
            params = CreateSandboxFromImageParams(
                image=Image.base("ubuntu:22.04"),
                resources=Resources(cpu=4, memory=16, disk=60, gpu=2),
            )
            asyncio.run(env._create_sandbox(params))
        finally:
            monkeypatch.undo()
        assert captured[0].resources.memory == expected_memory, package


def test_lite_config_lists_exactly_the_readme_lite_tasks():
    """harbor/run-daytona-lite.yaml is the README's 30-task table, verbatim.

    Users select MLS-Bench-Lite with ``-c run-daytona-lite.yaml``; this keeps
    the file, the README table and the rendered dataset from drifting apart.
    """
    import re

    readme = Path("README.md").read_text()
    section = re.search(r"## MLS-Bench-Lite(.*?)</details>", readme, re.S).group(1)
    table = {f"mls-bench__{tid}" for tid in re.findall(r"\]\(tasks/([a-z0-9\-]+)\)", section)}
    assert len(table) == 30

    lite = yaml.safe_load(Path("harbor/run-daytona-lite.yaml").read_text())
    full = yaml.safe_load(Path("harbor/run-daytona.yaml").read_text())
    (dataset,) = lite["datasets"]
    assert dataset["path"] == "tasks"
    names = dataset["task_names"]
    assert len(names) == len(set(names)) == 30
    assert set(names) == table
    for name in names:
        assert (Path("harbor/tasks") / name / "task.toml").is_file(), name
    # Same provider settings as the full-dataset config, so the two cannot drift.
    assert lite["environment"] == full["environment"]
    assert lite["n_concurrent_trials"] == full["n_concurrent_trials"]


def _h200_task(tmp_path: Path, name: str, gpus: int = 2) -> Path:
    """A rendered bundle with a two-entry h200 profile in tests/meta/config.json."""
    task_dir = tmp_path / name
    env_dir = task_dir / "environment"
    env_dir.mkdir(parents=True)
    (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    (env_dir / "docker-compose.yaml").write_text(_compose(gpus))
    meta = task_dir / "tests" / "meta"
    meta.mkdir(parents=True)
    (meta / "config.json").write_text(json.dumps({
        "seeds": [42],
        "test_cmds": [
            {"cmd": "scripts/train.sh", "compute": 2,
             "h200": {"compute": 1, "env": {"TP_SIZE": "1", "GPU_MEM_UTIL": "0.5"}}},
            {"cmd": "scripts/eval.sh", "compute": 1},
        ],
    }))
    return env_dir


def test_h200_exports_the_native_profile_env_to_the_sandbox(tmp_path: Path):
    """On Daytona the agent's shell sees the h200 block's env, not only the verifier.

    The native runner applies the block to the agent's own test runs; without
    the env the agent would explore with H100 settings on an H200 reservation.
    H100 requests export nothing but the type.
    """
    module = _module()
    from daytona import CreateSandboxFromImageParams, Image, Resources

    seen = {}
    for gpu_type in ("H200", "H100"):
        env_dir = _h200_task(tmp_path, f"task-{gpu_type}")
        trial_paths = TrialPaths(env_dir.parent / "trial")
        trial_paths.mkdir()
        env = module.DaytonaEnvironment(
            environment_dir=env_dir,
            environment_name=f"profile-{gpu_type}",
            session_id=f"profile-{gpu_type}.1",
            trial_paths=trial_paths,
            task_env_config=EnvironmentConfig(
                cpus=4, memory_mb=16384, storage_mb=61440, gpus=2
            ),
            gpu_type=gpu_type,
        )
        captured = []

        async def fake_parent_create_sandbox(self, params, *args, **kwargs):
            captured.append(params)
            self._sandbox = object()

        monkeypatch = pytest.MonkeyPatch()
        try:
            monkeypatch.setattr(
                module._HarborDaytonaEnvironment,
                "_create_sandbox",
                fake_parent_create_sandbox,
            )
            params = CreateSandboxFromImageParams(
                image=Image.base("ubuntu:22.04"),
                resources=Resources(cpu=4, memory=16, disk=60, gpu=2),
            )
            asyncio.run(env._create_sandbox(params))
        finally:
            monkeypatch.undo()
        seen[gpu_type] = captured[0].env_vars

    assert seen["H200"]["MLSBENCH_GPU_TYPE"] == "H200"
    assert seen["H200"]["TP_SIZE"] == "1"
    assert seen["H200"]["GPU_MEM_UTIL"] == "0.5"
    assert seen["H100"]["MLSBENCH_GPU_TYPE"] == "H100"
    assert "TP_SIZE" not in seen["H100"]


def test_docker_gpu_environment_takes_the_same_gpu_type_switch(tmp_path: Path):
    """Local Docker: ``--ek gpu_type=H200`` reaches every exec, agent and verifier."""
    module = _module()
    env_dir = _h200_task(tmp_path, "docker-task")
    trial_paths = TrialPaths(env_dir.parent / "trial")
    trial_paths.mkdir()
    config = EnvironmentConfig(cpus=4, memory_mb=16384, storage_mb=61440, gpus=2)

    plain = module.DockerGPUEnvironment(
        environment_dir=env_dir,
        environment_name="docker-plain",
        session_id="docker-plain.1",
        trial_paths=trial_paths,
        task_env_config=config,
    )
    assert plain.capabilities.gpus is True
    assert plain._merge_env(None) is None

    h200 = module.DockerGPUEnvironment(
        environment_dir=env_dir,
        environment_name="docker-h200",
        session_id="docker-h200.1",
        trial_paths=trial_paths,
        task_env_config=config,
        gpu_type="H200",
    )
    merged = h200._merge_env({"PER_EXEC": "1"})
    assert merged["MLSBENCH_GPU_TYPE"] == "H200"
    assert merged["TP_SIZE"] == "1"
    assert merged["PER_EXEC"] == "1"

    # harbor/harbor_env.py re-exports the same class for run.yaml.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "harbor_env_reexport", Path("harbor/harbor_env.py")
    )
    reexport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reexport)
    assert reexport.DockerGPUEnvironment is module.DockerGPUEnvironment


def test_gpu_sandboxes_repair_host_overlay_loader_paths_after_toolbox_ready(tmp_path: Path):
    """vLLM dlopens the libcudart path from /proc/self/maps verbatim; on runners
    that report host overlay paths the adapter aliases ``<prefix>/merged`` to
    ``/`` once the toolbox answers.  CPU sandboxes never run it."""
    module = _module()
    from daytona import CreateSandboxFromImageParams, Image, Resources

    ran = {}
    for gpus in (2, 0):
        env_dir = tmp_path / f"repair-{gpus}" / "environment"
        env_dir.mkdir(parents=True)
        (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")
        if gpus:
            (env_dir / "docker-compose.yaml").write_text(_compose(gpus))
        trial_paths = TrialPaths(env_dir.parent / "trial")
        trial_paths.mkdir()
        env = module.DaytonaEnvironment(
            environment_dir=env_dir,
            environment_name=f"repair-{gpus}",
            session_id=f"repair-{gpus}.1",
            trial_paths=trial_paths,
            task_env_config=EnvironmentConfig(
                cpus=4, memory_mb=16384, storage_mb=61440, gpus=gpus
            ),
        )
        commands = []

        async def fake_process_exec(command, timeout=None, **kwargs):
            commands.append(command)
            return SimpleNamespace(exit_code=0, result="aliased /var/lib/docker/overlay2/x/merged -> /")

        async def fake_parent_create_sandbox(self, params, *args, **kwargs):
            self._sandbox = SimpleNamespace(process=SimpleNamespace(exec=fake_process_exec))

        monkeypatch = pytest.MonkeyPatch()
        try:
            monkeypatch.setattr(
                module._HarborDaytonaEnvironment,
                "_create_sandbox",
                fake_parent_create_sandbox,
            )
            params = CreateSandboxFromImageParams(
                image=Image.base("ubuntu:22.04"),
                resources=Resources(cpu=4, memory=16, disk=60, gpu=gpus),
            )
            asyncio.run(env._create_sandbox(params))
        finally:
            monkeypatch.undo()
        ran[gpus] = commands

    assert ran[2][0] == "true"  # toolbox probe first
    assert any("/proc/self/maps" in c and "merged" in c for c in ran[2])
    assert ran[0] == ["true"]
