import numpy as np
from typing import Dict, Tuple, List, Callable
import random

from torch.utils.data import Dataset
import pandas as pd
import torch


class MergeGaussianDatasetGen(Dataset):
    def __init__(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        data_size: int = 100,
    ):
        self.nfolds = 10
        dict_data: Dict[List[np.ndarray], List[int]] = {"data": [], "label": []}

        for i, mui, sigmai in zip(range(len(mu)), mu, sigma):
            data = np.random.normal(mui, sigmai, data_size)
            dict_data["data"] += data.tolist()
            dict_data["label"] += [i] * data_size

        self.data = pd.DataFrame.from_dict(dict_data)

        self.split_data: Dict[str, List[np.ndarray]] = {
            "train": [],
            "validation": [],
        }
        self.split_labels: Dict[str, List[int]] = {"train": [], "validation": []}

    # ==========================================================
    @staticmethod
    def random_order(total_n: int) -> List[int]:

        indices = list(range(total_n))
        indices = random.sample(indices, len(indices))

        return indices

    # ===========================================================
    def split(self) -> None:

        indices = self.random_order(len(self.data))

        print(f"split dataset into train and validation with ratio = {self.nfolds}")

        split = len(self.data) // self.nfolds
        train_indices = indices[split:]
        print(train_indices)
        validation_indices = indices[:split]

        data_train = self.data.iloc[train_indices]
        self.split_data["train"] = list(data_train[["data"]].to_numpy())
        self.split_labels["train"] = list(data_train[["label"]].to_numpy())

        data_validation = self.data.iloc[validation_indices]
        self.split_data["validation"] = list(data_validation[["data"]].to_numpy())
        self.split_labels["validation"] = list(data_validation[["label"]].to_numpy())

    # =============================================================
    def get_data(self, train: bool) -> Tuple[List[np.ndarray], List[int]]:
        self.split()

        if train is True:
            sample = (self.split_data["train"], self.split_labels["train"])
        else:
            sample = (self.split_data["validation"], self.split_labels["validation"])

        return sample


# ============================================================================
class MergeGaussianDataset(Dataset):

    # ====================================================
    def __init__(
        self, data_gen: MergeGaussianDatasetGen, transform: Callable, train: bool
    ) -> None:
        self.transform = transform
        self.data, self.label = data_gen.get_data(train)
        assert len(self.data) == len(
            self.label
        ), " the number of data sould be the same with the number of label"

        # ====================================================

    def __len__(self) -> int:

        return len(self.data)

    # ====================================================
    def __getitem__(self, k: int) -> Tuple[torch.Tensor, int]:
        if self.transform is None:
            return self.data[k], self.label[k]
        return self.transform(self.data[k]), self.label[k]
