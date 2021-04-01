import torch
import torch.nn as nn


class FNN(nn.Module):
    """
    Feed forward neural network with two hidden layers, each layer contains 256 units. The initial
    learning rate is 1e-3, and reduces to 1e-3 every 20 epochs (starting at the 50th epochs)? In
    addition, for all hidden layers, dropout with ratio 0.5 and L2 weight decay 1e-2 is used. The
    model is trained with batch size 64 and MSE(MAE) as the loss function. Early stop is performed by
    monitoring the validation error.
    """
    def __init__(self, n_time):
        super(FNN, self).__init__()
        self.hid1 = nn.Linear(n_time, 256)
        self.hid2 = nn.Linear(256, 256)
        self.output = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor):
        x = x.squeeze(-1)  # [B, T, N, F=1] -> [B, T, N]
        x = x.transpose(1, 2)  # [B, T, N] -> [B, N, T]
        x = self.dropout(self.sigmoid(self.hid1(x)))
        x = self.dropout(self.sigmoid(self.hid2(x)))
        x = self.output(x)  # Output ~ [B, N, T=1]

        return x
