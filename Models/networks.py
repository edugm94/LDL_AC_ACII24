import sys
sys.path.append("..")
import torch
import torch.nn as nn
from utils.dl_utils import Linear, Chebynet, normalize_A
import torch.nn.functional as F
from utils.dl_utils import NewSGConv
import numpy as np
import math
from sklearn.neural_network import BernoulliRBM


class MLP(nn.Module):
    """
    Network architecture based on MLP to train on SEED/SEED5 dataset
    """
    def __init__(self, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_features=310, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.zeros_(m.bias.data)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        # out = self.relu(self.fc1(x))
        # out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out
