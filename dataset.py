import numpy as np
from typing import Dict, Tuple, List, Callable
import random
import seaborn as sns
from torch.utils.data import Dataset
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.model_selection import train_test_split

class MixGaussianGenerator():

    # ====================================================
    def __init__(
        self,
        mus: Tuple[np.ndarray, np.ndarray],
        sigmas: Tuple[float, float],
        data_dimension: int = 2,
        flag_plot: bool = True,
        data_size: int = 100,
    ) -> None:

        self.flag_plot=flag_plot
        self.data: List[float] = []
        self.label: List[int] = []
        # generate data
        np.random.seed(0)
        for i, mu, sigma in zip(range(2), mus, sigmas):
            assert (mu.size == data_dimension), f"size of mean :{mu.size()} not = {data_dimension}"
            data = np.random.multivariate_normal(mu, sigma * np.eye(data_dimension), data_size)
            self.data.extend(data.tolist())
            self.label.extend([i] * data_size)

        self.data = np.array(self.data)
        self.label = np.array(self.label).reshape(data_size * 2, -1)

    # ===========================================================
    def get_data(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[int], List[int]]:

        return train_test_split(self.data, self.label, random_state=0)

    # ===========================================================
    def plot_data(self)->None:
        if self.flag_plot is True :
            plt.plot(self.data[:,0],self.data[:,1],'x')
            plt.savefig("data_plot.png")


# ============================================================================
class MixGaussianDataset(Dataset):

    # ====================================================
    def __init__(
        self,
        data_generator :MixGaussianGenerator,
        train :bool,
    ) -> None:

        self.train = train
        self.data_train,self.data_test,self.label_train,self.label_test=data_generator.get_data()
        assert len(self.data_train) == len(self.label_train), f"data size {len(self.data_train)} != label size{len(self.label_train)}"
        assert len(self.data_test) == len(self.label_test), f"data size {len(self.data_test)}!= label size{len(self.label_test)}"
    # ===========================================================
    def __len__(self) -> int:

        if self.train is True:
            return len(self.data_train)
        else:

            return len(self.data_test)

    # ====================================================
    def __getitem__(self, k: int) -> Tuple[np.ndarray, int]:

        if self.train is True:

            return self.data_train[k], self.label_train[k]
        else:

            return self.data_test[k], self.label_test[k]
