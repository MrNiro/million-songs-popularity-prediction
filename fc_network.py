import torch
import torch.nn as nn
import torch.nn.functional as F


class FC_Net(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()

        self.hidden1 = nn.Linear(input_dims, 2048)
        # self.hidden2 = nn.Linear(4096, 2048)
        self.hidden3 = nn.Linear(2048, 1024)
        self.hidden4 = nn.Linear(1024, 256)
        self.hidden5 = nn.Linear(256, 128)
        self.output = nn.Linear(128, output_dims)
        self.dropout = nn.Dropout(0.2)

    def forward(self, features):
        middle = self.hidden1(features)
        middle = F.relu6(middle)
        middle = self.dropout(middle)

        # middle = self.hidden2(middle)
        # middle = F.relu6(middle)
        # middle = self.dropout(middle)

        middle = self.hidden3(middle)
        middle = F.relu6(middle)
        middle = self.dropout(middle)

        middle = self.hidden4(middle)
        middle = F.relu6(middle)
        middle = self.dropout(middle)

        middle = self.hidden5(middle)
        middle = F.relu6(middle)
        middle = self.dropout(middle)

        out = self.output(middle)
        out = torch.sigmoid(out)
        return out
