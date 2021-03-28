import torch
import torch.nn as nn


class FNN(nn.Module):
    """
    Feed forward neural network with two hidden layers, each layer contains 256 units. The initial
    learning rate is 1e-3, and reduces to 1e-3 every 20 epochs starting at the 50th epochs. In
    addition, for all hidden layers, dropout with ratio 0.5 and L2 weight decay 1e-2 is used. The
    model is trained with batch size 64 and MAE as the loss function. Early stop is performed by
    monitoring the validation error.
    """
    def __init__(self):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(1, 256)
        self.fc2 = nn.Linear(256, 256)
        self.output = nn.Linear(256, 1)

    def forward(self):
        pass
