import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Custom model for time series imputation.

    Forward signature: forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)
    - x_enc: [batch, seq_len, enc_in] — input with masked values set to 0
    - x_mark_enc: [batch, seq_len, time_features] — time feature encoding
    - x_dec: not used for imputation (None)
    - x_mark_dec: not used for imputation (None)
    - mask: [batch, seq_len, enc_in] — binary mask (1=observed, 0=masked)

    Must return: [batch, seq_len, enc_in] — reconstructed sequence
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.seq_len  # For imputation, pred_len = seq_len
        self.enc_in = configs.enc_in
        # TODO: Define your model architecture here

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        """
        Imputation: reconstruct missing values in the input sequence.
        Input: x_enc [batch, seq_len, enc_in] with zeros at masked positions
        Mask: [batch, seq_len, enc_in], 1=observed, 0=masked
        Output: [batch, seq_len, enc_in]
        """
        # TODO: Implement your imputation logic
        return x_enc  # Placeholder: return input as-is

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out
        return None
