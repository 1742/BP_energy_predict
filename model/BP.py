import torch
from torch import nn


class BP(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(BP, self).__init__()
        self.input = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=64, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.hidden_1 = nn.Sequential(
            nn.Linear(in_features=64, out_features=128, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.hidden_2 = nn.Sequential(
            nn.Linear(in_features=128, out_features=256, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        self.hidden_3 = nn.Sequential(
            nn.Linear(in_features=256, out_features=512, bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
        self.hidden_4 = nn.Sequential(
            nn.Linear(in_features=512, out_features=1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True)
        )
        self.hidden_5 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=2048, bias=True),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True)
        )
        self.hidden_6 = nn.Sequential(
            nn.Linear(in_features=2048, out_features=4096, bias=True),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True)
        )
        self.out = nn.Linear(in_features=4096, out_features=out_features, bias=True)

    def forward(self, t, x):
        x = torch.cat((t, x), dim=1)

        x = self.input(x)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        x = self.hidden_3(x)
        x = self.hidden_4(x)
        x = self.hidden_5(x)
        x = self.hidden_6(x)
        x = self.out(x)

        return x


if __name__ == '__main__':
    mdoel = BP(102, 96)
    print(mdoel)
