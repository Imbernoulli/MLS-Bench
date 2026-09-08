import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Custom model for time series classification.

    Forward signature: forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)
    - x_enc: [batch, seq_len, enc_in] — input time series
    - x_mark_enc: [batch, seq_len] — padding mask (1=valid, 0=padding)
    - x_dec: not used (None)
    - x_mark_dec: not used (None)

    Must return: [batch, num_class] — class logits (before softmax)

    Note: configs.seq_len, configs.enc_in, and configs.num_class are set
    dynamically from the dataset at runtime.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.num_class = configs.num_class
        # TODO: Define your model architecture here

    def classification(self, x_enc, x_mark_enc):
        """
        Classification: assign a label to the input time series.
        Input: x_enc [batch, seq_len, enc_in]
        x_mark_enc: [batch, seq_len] padding mask
        Output: [batch, num_class] logits
        """
        # TODO: Implement your classification logic
        batch_size = x_enc.shape[0]
        return torch.zeros(batch_size, self.num_class).to(x_enc.device)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out
        return None
