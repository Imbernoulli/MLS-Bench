import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Custom model for time series anomaly detection.

    Forward signature: forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)
    - x_enc: [batch, seq_len, enc_in] — input time series
    - x_mark_enc: not used for anomaly detection (None)
    - x_dec: not used for anomaly detection (None)
    - x_mark_dec: not used for anomaly detection (None)

    Must return: [batch, seq_len, c_out] — reconstructed sequence
    The framework computes MSE between input and output for anomaly scoring.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.seq_len  # For anomaly detection, pred_len = seq_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        # TODO: Define your model architecture here

    def anomaly_detection(self, x_enc):
        """
        Anomaly detection: reconstruct the input sequence.
        Input: x_enc [batch, seq_len, enc_in]
        Output: [batch, seq_len, c_out]
        """
        # TODO: Implement your reconstruction logic
        return x_enc  # Placeholder: identity reconstruction

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out
        return None
