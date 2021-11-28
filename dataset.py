import numpy as np
from typing import Tuple, List
from torch.utils.data import Dataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class MixGaussianGenerator:

    # ====================================================
    def __init__(
        self,
        mus: Tuple[np.ndarray, np.ndarray],
        sigmas: Tuple[float, float],
        data_dimension: int = 3,
        flag_plot: bool = True,
        data_size: int = 300,
    ) -> None:

        self.flag_plot = flag_plot
        data_tmp: List[List[float]] = []
        label_tmp: List[int] = []
        # generate data
        np.random.seed(0)
        for i, (mu, sigma) in enumerate(zip(mus, sigmas)):
            assert (
                mu.size == data_dimension
            ), f"size of mean :{mu.size()} not = {data_dimension}"
            data = np.random.multivariate_normal(
                mu, sigma * np.eye(data_dimension), data_size
            )
            data_tmp.extend(data.tolist())
            label_tmp.extend([i] * data_size)
        self.data = np.array(data_tmp)
        self.label = np.array(label_tmp).reshape(data_size * 2, -1)
        if flag_plot is True:
            self.plot_data()

    # ===========================================================
    def get_data(
        self,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[int], List[int]]:

        return train_test_split(self.data, self.label, random_state=0)

    # ===========================================================
    def plot_data(self) -> None:
        concat_data_label = np.concatenate((self.data[:, :2], self.label), axis=1)
        df = pd.DataFrame(data=concat_data_label, columns=["data0", "data1", "label"])
        sns.scatterplot(data=df, x="data0", y="data1", hue="label")
        plt.savefig("data_plot.png")


# ============================================================================
class MixGaussianDataset(Dataset):

    # ====================================================
    def __init__(
        self,
        data_generator: MixGaussianGenerator,
        train: bool,
    ) -> None:

        self.train = train
        (
            self.data_train,
            self.data_test,
            self.label_train,
            self.label_test,
        ) = data_generator.get_data()
        assert len(self.data_train) == len(
            self.label_train
        ), f"data size {len(self.data_train)} != label size{len(self.label_train)}"
        assert len(self.data_test) == len(
            self.label_test
        ), f"data size {len(self.data_test)}!= label size{len(self.label_test)}"

    # ===========================================================
    def __len__(self) -> int:

        if self.train:
            return len(self.data_train)
        else:

            return len(self.data_test)

    # ====================================================
    def __getitem__(self, k: int) -> Tuple[np.ndarray, int]:

        if self.train:

            return self.data_train[k], self.label_train[k]
        else:

            return self.data_test[k], self.label_test[k]
