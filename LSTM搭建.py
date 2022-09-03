import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm_1 = nn.LSTM(1, 16, 2, batch_first=True)
        self.lstm_2 = nn.LSTM(16, 32, 2, batch_first=True)
        self.fc_1 = nn.Linear(32, 16)
        self.fc_2 = nn.Linear(16, 1)

    def forward(self, x):
        out, (hn, cn) = self.lstm_1(x)
        out_1 = self.fc_2(out[:, -1, :])

        return out_1
