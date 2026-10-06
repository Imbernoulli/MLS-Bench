import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Custom model for exogenous variable forecasting (features=MS).

    Forward signature: forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)
    - x_enc: [batch, seq_len, enc_in] — all input variables
    - x_mark_enc: [batch, seq_len, time_features] — time feature encoding
    - x_dec: [batch, label_len+pred_len, dec_in] — decoder input
    - x_mark_dec: [batch, label_len+pred_len, time_features] — decoder time features

    Must return: [batch, pred_len, c_out] for forecasting
    Note: c_out = enc_in. The framework extracts the target (last dim) for MS mode.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        # TODO: Define your model architecture here

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Forecasting with exogenous variables.
        Input: x_enc [batch, seq_len, enc_in] — all variables
        Output: [batch, pred_len, c_out] — predict all variables
        """
        # TODO: Implement your forecasting logic
        batch_size = x_enc.shape[0]
        return torch.zeros(batch_size, self.pred_len, self.c_out).to(x_enc.device)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None
