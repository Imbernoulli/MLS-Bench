import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional

from basicts.configs import BasicTSModelConfig


@dataclass
class CustomConfig(BasicTSModelConfig):
    """Configuration for the Custom spatial-temporal forecasting model.

    Required fields (set by training script):
        input_len: Length of input historical sequence.
        output_len: Length of output prediction sequence.
        num_features: Number of spatial nodes (sensors).

    Optional fields (tunable):
        hidden_size: Hidden dimension size.
        num_layers: Number of model layers.
        dropout: Dropout rate.
    """

    input_len: int = field(default=12, metadata={"help": "Input sequence length."})
    output_len: int = field(default=12, metadata={"help": "Output sequence length."})
    num_features: int = field(default=207, metadata={"help": "Number of spatial nodes."})
    hidden_size: int = field(default=64, metadata={"help": "Hidden dimension size."})
    num_layers: int = field(default=2, metadata={"help": "Number of model layers."})
    dropout: float = field(default=0.1, metadata={"help": "Dropout rate."})


class Custom(nn.Module):
    """
    Custom model for spatial-temporal traffic forecasting.

    The model receives traffic measurements from N spatial nodes over T time steps
    and predicts the next T' time steps for all N nodes.

    Forward signature: forward(inputs, inputs_timestamps)
    - inputs: [batch_size, input_len, num_features] — historical traffic data
      Each feature dimension corresponds to one spatial node (sensor).
    - inputs_timestamps: [batch_size, input_len, num_timestamps] — temporal features
      Typically contains normalized time-of-day and day-of-week.

    Must return: [batch_size, output_len, num_features] — predicted traffic values
    """

    def __init__(self, config: CustomConfig):
        super().__init__()
        self.input_len = config.input_len
        self.output_len = config.output_len
        self.num_features = config.num_features
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        # TODO: Define your model architecture here

    def forward(self, inputs: torch.Tensor, inputs_timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [batch_size, input_len, num_features] — historical traffic data
            inputs_timestamps: [batch_size, input_len, num_timestamps] — time features

        Returns:
            prediction: [batch_size, output_len, num_features]
        """
        # TODO: Implement your spatial-temporal forecasting model
        # Placeholder: simple linear projection (no spatial modeling)
        batch_size = inputs.shape[0]
        return torch.zeros(batch_size, self.output_len, self.num_features, device=inputs.device)


# CONFIG_OVERRIDES: override training hyperparameters for your method.
# Allowed keys: lr, weight_decay.
CONFIG_OVERRIDES = {}
