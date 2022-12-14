import torch
import torch.nn as nn
import torch.nn.functional as F


class FC_Net(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()

        # input dims is the same as Features Number
        self.hidden1 = nn.Linear(input_dims, 256)
        self.hidden2 = nn.Linear(4096, 2048)
        self.hidden3 = nn.Linear(2048, 1024)
        self.hidden4 = nn.Linear(1024, 256)
        self.hidden5 = nn.Linear(256, 128)
        self.output = nn.Linear(128, output_dims)

        # Batch Normalization for each layer
        self.batch_norm_init = nn.BatchNorm1d(input_dims)
        self.batch_norm_1 = nn.BatchNorm1d(256)
        self.batch_norm_2 = nn.BatchNorm1d(2048)
        self.batch_norm_3 = nn.BatchNorm1d(1024)
        self.batch_norm_4 = nn.BatchNorm1d(256)
        self.batch_norm_5 = nn.BatchNorm1d(128)

        # Dropout for each layer
        self.dropout = nn.Dropout(0.4)

    def forward(self, features):
        features = self.batch_norm_init(features)

        middle = self.hidden1(features)
        middle = F.relu6(middle)
        middle = self.batch_norm_1(middle)
        middle = self.dropout(middle)

        # middle = self.hidden2(middle)
        # middle = F.relu6(middle)
        # middle = self.batch_norm_2(middle)
        # middle = self.dropout(middle)
        #
        # middle = self.hidden3(middle)
        # middle = F.relu6(middle)
        # middle = self.batch_norm_3(middle)
        # middle = self.dropout(middle)
        #
        # middle = self.hidden4(middle)
        # middle = F.relu6(middle)
        # middle = self.batch_norm_4(middle)
        # middle = self.dropout(middle)

        middle = self.hidden5(middle)
        middle = F.relu6(middle)
        middle = self.batch_norm_5(middle)
        middle = self.dropout(middle)

        out = self.output(middle)
        out = torch.sigmoid(out)
        return out
