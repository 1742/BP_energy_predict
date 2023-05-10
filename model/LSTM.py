import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(1), self.hidden_size)
        c0 = torch.zeros(1, x.size(1), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[-1, :, :])
        return out


if __name__ == '__main__':
    series = torch.rand((1, 1, 3))
    mdoel = LSTM(3, 50, 96)
    print(mdoel)
    out = mdoel(series)
    loss = nn.MSELoss(out, series)
    print(out)
    print(loss)