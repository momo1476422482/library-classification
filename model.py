import numpy as np
from torch import nn
import torch


class MLP(nn.Module):
    # define model elements
    def __init__(self, n_inputs: int = 2):
        super().__init__()
        self.layer = nn.Linear(n_inputs, 1, bias=True)
        self.activation = nn.Sigmoid()

    # forward propagate input
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.layer(X)
        p = self.activation(X)
        return torch.cat((p, 1 - p), 1)
