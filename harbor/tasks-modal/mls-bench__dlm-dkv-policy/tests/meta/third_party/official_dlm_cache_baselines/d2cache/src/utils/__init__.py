"""Lightweight utility exports for the task-local d2Cache subset."""

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from .common import *


def is_adapted_from_ar(model_or_config: PreTrainedModel | PretrainedConfig) -> bool:
    """Official d2Cache utility, trimmed to avoid evaluation-only dependencies."""
    if isinstance(model_or_config, PreTrainedModel):
        config = model_or_config.config
    elif isinstance(model_or_config, PretrainedConfig):
        config = model_or_config
    else:
        raise ValueError(
            "Expected model_or_config to be a PreTrainedModel or PretrainedConfig, "
            f"got {type(model_or_config)}"
        )

    model_type = str(config.model_type).lower()
    if model_type == "llada":
        return False
    if model_type == "dream":
        return True
    raise ValueError(f"Unsupported model type: {config.model_type}")
