import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Linear, self).__init__()
        self.fc0 = nn.Linear(n_input, n_hidden)
        self.relu0 = nn.ReLU(inplace=True)
        self.block1 = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.5),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.5),
        )
        self.block2 = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.5),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.5),
        )
        self.fc3 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.fc0(x)
        x = self.relu0(x)
        x1 = self.block1(x)
        x1 += x
        x2 = self.block2(x1)
        x2 += x1
        y = self.fc3(x2)
        return y