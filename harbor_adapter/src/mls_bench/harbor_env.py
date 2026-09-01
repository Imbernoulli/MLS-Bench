"""Harbor environments used by the MLS-Bench Daytona run.

The rendered MLS-Bench tasks contain a small Docker Compose override for every
GPU task.  The override is useful with local Docker (it asks Docker to expose
the requested NVIDIA devices), but it is *not* a real multi-container task.
Harbor's Daytona provider quite correctly runs real Compose definitions through
Docker-in-Docker and, in recent releases, rejects GPU+Compose because the
inner container cannot see the Daytona GPU.

``DaytonaEnvironment`` below recognises this adapter-specific, GPU-only
override and runs the task as a direct Daytona GPU sandbox instead.  CPU tasks
and genuine Compose tasks continue to use Harbor's normal provider behaviour.
The small ``Resources`` compatibility shim fills the GPU field on Harbor
0.6.x, whose Daytona implementation omitted it; on newer Harbor releases the
provider already builds the resource object correctly and the shim is a no-op.
"""

from __future__ import annotations

import contextvars
import copy
import importlib
import os
from pathlib import Path
from typing import Any

import yaml

from harbor.environments.capabilities import EnvironmentCapabilities

try:  # Daytona >=0.205 exposes explicit GPU type selection.
    from daytona import GpuType as _DaytonaGpuType
except ImportError:  # pragma: no cover - only exercised with old SDKs
    _DaytonaGpuType = None


# Harbor 0.6.x keeps the provider in a module; newer Harbor releases moved it
# into the ``harbor.environments.daytona`` package.  Import the implementation
# module first and resolve private strategy classes from there.  Some releases
# re-export those classes from the package while others do not, so importing
# them directly from ``harbor.environments.daytona`` is not portable.
try:  # Harbor >= 0.7/current
    _DAYTONA_IMPL = importlib.import_module("harbor.environments.daytona.environment")
except (ImportError, ModuleNotFoundError):  # Harbor 0.6.x
    _DAYTONA_IMPL = importlib.import_module("harbor.environments.daytona")

try:
    _HarborDaytonaEnvironment = _DAYTONA_IMPL.DaytonaEnvironment
    _DaytonaDirect = _DAYTONA_IMPL._DaytonaDirect
    _DaytonaDinD = _DAYTONA_IMPL._DaytonaDinD
except AttributeError as exc:  # pragma: no cover - provider-version diagnostic
    # A few intermediate releases put the classes in the package's
    # ``__init__`` while retaining an ``environment`` helper module.
    _package = importlib.import_module("harbor.environments.daytona")
    try:
        _HarborDaytonaEnvironment = _package.DaytonaEnvironment
        _DaytonaDirect = _package._DaytonaDirect
        _DaytonaDinD = _package._DaytonaDinD
        _DAYTONA_IMPL = _package
    except AttributeError:
        raise ImportError(
            "Installed Harbor Daytona provider does not expose its strategy classes"
        ) from exc

# Expose the manager for diagnostics/tests and for callers that need to reset
# the provider singleton between isolated jobs.  Resolve it from the package
# as a fallback for releases that keep it out of ``environment.py``.
try:
    DaytonaClientManager = _DAYTONA_IMPL.DaytonaClientManager
except AttributeError:  # pragma: no cover - provider-version diagnostic
    DaytonaClientManager = importlib.import_module(
        "harbor.environments.daytona"
    ).DaytonaClientManager


_GPU_CONTEXT: contextvars.ContextVar[int] = contextvars.ContextVar(
    "mlsbench_daytona_gpu_count", default=0
)
# Strategies may be re-exported by a package even though their globals (and
# therefore ``Resources``) live in a different implementation module.
_RESOURCES_IMPL = importlib.import_module(_DaytonaDirect.__module__)
_ORIGINAL_RESOURCES = getattr(_RESOURCES_IMPL, "Resources", None)


if _ORIGINAL_RESOURCES is not None:

    class _GPUAwareResources:
        """Add the task GPU count when running Harbor 0.6.x strategies.

        Harbor's old ``_DaytonaDirect`` and ``_DaytonaDinD`` construct
        ``Resources(cpu=..., memory=..., disk=...)`` themselves.  A context
        variable keeps this compatibility layer safe when Harbor starts trials
        concurrently; newer providers either pass ``gpu`` already or do not
        construct this object in the strategy.
        """

        def __new__(cls, *args: Any, **kwargs: Any) -> Any:
            gpu_count = _GPU_CONTEXT.get()
            if gpu_count > 0 and "gpu" not in kwargs:
                kwargs["gpu"] = gpu_count
            return _ORIGINAL_RESOURCES(*args, **kwargs)

    # The strategy resolves Resources from its implementation module globals.
    # Install the proxy once; calls made outside our context are byte-for-byte
    # equivalent to the original class.
    if not getattr(_RESOURCES_IMPL, "_MLSBENCH_GPU_RESOURCES", False):
        _RESOURCES_IMPL.Resources = _GPUAwareResources
        _RESOURCES_IMPL._MLSBENCH_GPU_RESOURCES = True


class _GpuAwareStrategyMixin:
    async def start(self, force_build: bool):  # type: ignore[no-untyped-def]
        token = _GPU_CONTEXT.set(max(0, int(self._env.task_env_config.gpus or 0)))
        try:
            return await super().start(force_build)  # type: ignore[misc]
        finally:
            _GPU_CONTEXT.reset(token)


class _GpuAwareDirect(_GpuAwareStrategyMixin, _DaytonaDirect):
    pass


class _GpuAwareDinD(_GpuAwareStrategyMixin, _DaytonaDinD):
    pass


def is_gpu_reservation_only_compose(
    environment_dir: Path,
    *,
    gpu_count: int,
) -> bool:
    """Return whether ``docker-compose.yaml`` only reserves NVIDIA devices.

    This deliberately accepts a narrow schema.  A compose file with another
    service, a build/image override, or any unknown key remains a genuine
    Compose task and is delegated to Harbor (which will report the provider's
    normal GPU+Compose limitation rather than silently dropping a service).
    """

    if gpu_count <= 0:
        return False
    return _is_gpu_reservation_only_compose(
        environment_dir,
        expected_gpu_count=gpu_count,
    )


def _is_gpu_reservation_only_compose(
    environment_dir: Path,
    *,
    expected_gpu_count: int | None,
) -> bool:
    """Validate the reservation-only schema, optionally checking its count.

    ``expected_gpu_count=None`` is used only when Harbor's explicit
    ``override_gpus`` changes the effective allocation.  In that case the
    compose overlay's original count is intentionally different from the
    Daytona request, but the overlay is still safe to ignore because it has no
    service semantics beyond a device reservation.
    """

    path = environment_dir / "docker-compose.yaml"
    if not path.is_file():
        return False
    try:
        document = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError):
        return False
    if not isinstance(document, dict):
        return False
    services = document.get("services")
    if not isinstance(services, dict) or set(services) != {"main"}:
        return False
    main = services.get("main")
    if not isinstance(main, dict) or set(main) != {"deploy"}:
        return False
    deploy = main.get("deploy")
    if not isinstance(deploy, dict) or set(deploy) != {"resources"}:
        return False
    resources = deploy.get("resources")
    if not isinstance(resources, dict) or set(resources) != {"reservations"}:
        return False
    reservations = resources.get("reservations")
    if not isinstance(reservations, dict) or set(reservations) != {"devices"}:
        return False
    devices = reservations.get("devices")
    if not isinstance(devices, list) or len(devices) != 1:
        return False
    device = devices[0]
    if not isinstance(device, dict):
        return False
    if set(device) != {"driver", "count", "capabilities"}:
        return False
    count = device.get("count")
    if (
        device.get("driver") != "nvidia"
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
    ):
        return False
    if expected_gpu_count is not None and count != expected_gpu_count:
        return False
    capabilities = device.get("capabilities")
    return isinstance(capabilities, list) and "gpu" in capabilities


class DaytonaEnvironment(_HarborDaytonaEnvironment):
    """Daytona provider with MLS-Bench's GPU-only Compose compatibility."""

    def _spot_requested(self) -> bool:
        """Return whether this run requested a Daytona spot sandbox.

        Harbor forwards ``--ek spot=true`` as an environment kwarg.  The
        upstream Harbor Daytona provider currently does not forward that
        kwarg to Daytona's ``CreateSandbox*Params`` objects, so the adapter
        applies it at the final sandbox-creation boundary below.  Daytona
        rejects ``spot`` for CPU sandboxes; callers therefore only enable it
        when the effective task resource request includes one or more GPUs.
        """
        value = self._kwargs.get("spot", False)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _gpu_type_requested(self):  # type: ignore[no-untyped-def]
        """Resolve an optional Daytona GPU type kwarg (e.g. ``H200``)."""
        # Hopper is the portable default for the prebuilt MLS-Bench images:
        # their CUDA wheels support H100/H200 (sm_90) but older images do not
        # support Daytona's Blackwell RTX PRO 6000 (sm_120).
        value = self._kwargs.get(
            "gpu_type", "H200" if self._spot_requested() else None
        )
        if value is None or _DaytonaGpuType is None:
            return None
        name = str(value).strip().upper().replace("-", "_")
        # Accept the SDK enum names and common CLI spellings while rejecting
        # arbitrary values before they reach the provider API.
        aliases = {
            "RTX_PRO_6000_BLACKWELL": "RTX_PRO_6000",
            "RTX_PRO_6000_BLACKWELL_SERVER": "RTX_PRO_6000",
            "RTX_4090": "RTX_4090",
            "4090": "RTX_4090",
            "RTX_5090": "RTX_5090",
            "5090": "RTX_5090",
        }
        name = aliases.get(name, name)
        try:
            return _DaytonaGpuType[name]
        except (KeyError, TypeError):
            raise ValueError(f"Unsupported Daytona gpu_type={value!r}")

    async def _create_sandbox(self, params, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        """Forward optional spot/GPU-type flags to Daytona's SDK request.

        This hook is shared by Harbor's direct and DinD strategies, including
        the adapter's GPU-only Compose compatibility strategy. Keeping the
        change here avoids duplicating the provider's resource construction and
        preserves the task's declared GPU count verbatim. GPU type selection
        also works for on-demand requests; spot remains opt-in.
        """
        gpu_count = int(self.task_env_config.gpus or 0)
        gpu_type = self._gpu_type_requested()
        if gpu_count > 0:
            if self._spot_requested() and hasattr(params, "spot"):
                params.spot = True
            resources = getattr(params, "resources", None)
            if resources is not None and gpu_type is not None:
                resources.gpu_type = gpu_type
            # The verifier runs inside the Daytona sandbox (and, for GPU-only
            # overlays, the direct strategy). Propagate the selected device
            # type explicitly so its H200-aware scheduler does not depend on
            # whether the image happens to include nvidia-smi.
            if gpu_type is not None and hasattr(params, "env_vars"):
                env_vars = dict(getattr(params, "env_vars", None) or {})
                env_vars["MLSBENCH_GPU_TYPE"] = getattr(
                    gpu_type, "value", str(gpu_type)
                )
                params.env_vars = env_vars
        # The smoke runner sets a per-invocation label so its optional orphan
        # cleanup can never delete another runner's organization sandbox.
        # Keep this best-effort for older Daytona SDKs that lack ``labels``.
        run_id = os.environ.get("MLSBENCH_DAYTONA_RUN_ID")
        if run_id and hasattr(params, "labels"):
            labels = dict(getattr(params, "labels", None) or {})
            labels["mlsbench-run-id"] = run_id
            params.labels = labels
        return await super()._create_sandbox(params, *args, **kwargs)

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        # Keep all capabilities supplied by the installed Harbor version and
        # advertise GPU allocation (the 0.6.x provider forgot this flag).
        base = super().capabilities
        if hasattr(base, "model_copy"):
            return base.model_copy(update={"gpus": True})
        values = base.model_dump() if hasattr(base, "model_dump") else base.dict()
        values["gpus"] = True
        return EnvironmentCapabilities(**values)

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: Any,
        task_env_config: Any,
        *args: Any,
        **kwargs: Any,
    ):
        declared_gpus = int(getattr(task_env_config, "gpus", 0) or 0)
        override_gpus = kwargs.get("override_gpus")
        requested_gpus = (
            int(override_gpus) if override_gpus is not None else declared_gpus
        )
        extra_compose = kwargs.get("extra_docker_compose")
        gpu_only = (
            not extra_compose
            and declared_gpus > 0
            and (
                is_gpu_reservation_only_compose(
                    environment_dir,
                    gpu_count=declared_gpus,
                )
                or (
                    override_gpus is not None
                    and _is_gpu_reservation_only_compose(
                        environment_dir,
                        expected_gpu_count=None,
                    )
                )
            )
        )

        # A Daytona spot GPU is useful only when the task itself runs in the
        # GPU sandbox. A genuine Compose definition is executed through
        # Docker-in-Docker; its inner containers cannot see the outer
        # Daytona GPU, so accepting spot here would spend scarce capacity on
        # a run that can never satisfy the task's GPU contract.
        spot_value = kwargs.get("spot", False)
        if isinstance(spot_value, str):
            spot_value = spot_value.strip().lower() in {"1", "true", "yes", "on"}
        if bool(spot_value) and declared_gpus > 0 and not gpu_only:
            raise ValueError(
                "Daytona spot GPU requires a direct GPU sandbox; "
                "genuine multi-container Compose tasks are unsupported"
            )

        # Let Harbor run its complete normal initialisation with a zero-GPU
        # clone, then switch this adapter-only override to the direct strategy
        # and restore the effective (possibly overridden) GPU count before any
        # lifecycle call.  The original model is not mutated, which is
        # important when Harbor reuses task config data across trials.
        effective_config = (
            copy.deepcopy(task_env_config) if gpu_only else task_env_config
        )
        if gpu_only:
            effective_config.gpus = 0

        # Newer Harbor releases validate ``override_gpus`` while they still
        # see the compose definition and reject GPU+Compose before this
        # adapter can switch to the direct strategy.  The effective config is
        # already zero-GPU during parent initialization, so omit the override
        # kwarg there and restore the requested count below.
        parent_kwargs = dict(kwargs)
        if gpu_only:
            parent_kwargs.pop("override_gpus", None)

        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=effective_config,
            *args,
            **parent_kwargs,
        )

        if gpu_only:
            self.task_env_config.gpus = requested_gpus
            # Newer Harbor caches effective GPU values only through the task
            # model property, so restoring the model is sufficient.
            self._compose_mode = False
            self._strategy = _GpuAwareDirect(self)
        elif not getattr(self, "_compose_mode", False) and requested_gpus > 0:
            # Covers direct GPU tasks authored outside this adapter on Harbor
            # 0.6.x, whose strategy omitted ``Resources.gpu``.
            self._strategy = _GpuAwareDirect(self)


__all__ = [
    "DaytonaEnvironment",
    "DaytonaClientManager",
    "is_gpu_reservation_only_compose",
]
