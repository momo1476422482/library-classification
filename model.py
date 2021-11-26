from typing import Optional
import numpy as np
from sklearn.svm import LinearSVC
from torch import nn
from torchvision import transforms


class ModelClassifSup:
    def __init__(self) -> None:
        self.model = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    def infer(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ModelSVM(ModelClassifSup):
    def __init__(self, random_state: Optional[int] = 0, tol: float = 1e-5) -> None:
        super().__init__()

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.model is None:
            X = X.reshape(len(X), -1)
            y = y.reshape(len(y), -1)
            self.model = LinearSVC(random_state=0, tol=1e-5)
            self.model.fit(X, y)

    def infer(self, X: np.ndarray) -> np.ndarray:
        X = X.reshape(len(X), -1)
        return self.model.predict(X)


class MLP(nn.Module):
    # define model elements
    def __init__(self, n_inputs: int = 2):
        super().__init__()
        self.layer = nn.Linear(n_inputs, 2,bias=True)
        self.activation = nn.Sigmoid()

    # forward propagate input
    def forward(self, X: np.ndarray) -> float:
        X = self.layer(X)
        X = self.activation(X)

        return X