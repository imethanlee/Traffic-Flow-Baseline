import torch
import torch.nn as nn


class FC_LSTM(nn.Module):
    """
    The Encoder-decoder framework using LSTM with peephole (Sutskever et al., 2014).
    Both the encoder and the decoder contain two recurrent layers. In each recurrent layer,
    there are 256 LSTM units, L1 weight decay is 2e-5, L2 weight decay 5e-4. The model is
    trained with batch size 64 and loss function MAE. The initial learning rate is 1e-4 and
    reduces to 1/10 every 10 epochs starting from the 20th epochs. Early stop is performed by
    monitoring the validation error.
    """
    def __init__(self):
        super(FC_LSTM, self).__init__()

    def forward(self):
        pass
