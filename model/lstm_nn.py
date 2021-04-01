import torch
import torch.nn as nn


class LSTM_NN(nn.Module):
    """
    This is an implementation of "Long short-term memory neural network for traffic speed
    prediction using remote microwave sensor data"
    """
    def __init__(self):
        super(LSTM_NN, self).__init__()
        self.n_hid = 256
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.n_hid, num_layers=1, batch_first=True)
        self.output = nn.Linear(self.n_hid, 1)

    def forward(self, x):
        _, n_time, n_node, n_feat = x.shape
        x = x.transpose(1, 2)  # [B, T, N, F] -> [B, N, T, F]
        x = x.reshape(-1, n_time, n_feat)  # [B, N, T, F] -> [B * N, T, F]
        _, (hn, _) = self.lstm(x)
        hn = hn.reshape(1, -1, n_node, self.n_hid).transpose(0, 1)
        return self.output(hn).contiguous()  # Output ~ [B, N, T=1]
