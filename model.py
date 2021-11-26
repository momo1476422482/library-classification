from typing import Optional
import numpy as np
from sklearn.svm import LinearSVC
from torch import nn
from torchvision import transforms


class MLP(nn.Module):
    # define model elements
    def __init__(self, n_inputs: int = 2):
        super().__init__()
        self.layer = nn.Linear(n_inputs, 2, bias=True)
        self.activation = nn.Sigmoid()

    # forward propagate input
    def forward(self, X: np.ndarray) -> float:
        X = self.layer(X)
        X = self.activation(X)

        return X
