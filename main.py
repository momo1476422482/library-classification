import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import MergeGaussianDatasetGen, MergeGaussianDataset
import numpy as np
from model import ModelSVM
import sys
from typing import Optional


def eval_algo(algo: str) -> None:

    data_generator = MergeGaussianDatasetGen([0, 1], [0.1, 0.2])
    train_set = MergeGaussianDataset(
        data_generator,
        train=True,
        transform=None,
    )
    print(train_set[10])

    # train model
    train_loader = DataLoader(train_set, batch_size=len(train_set))
    x, y = next(iter(train_loader))

    if algo.lower() == "svmlin":

        model = ModelSVM()

    print(f"run {algo.upper()}")
    model.train(x, y)

    # test model
    test_set = MergeGaussianDataset(data_generator, train=False, transform=None)

    test_loader = DataLoader(test_set, batch_size=len(test_set))
    x_test, y_test = next(iter(test_loader))
    # compute score
    with torch.no_grad():
        score_test = model.infer(x_test)

    # compute the error between predicted label and the real one
    error = np.abs(y_test - score_test)
    print("error", error)


def main():

    if len(sys.argv) == 2:
        algo = sys.argv[1]
    else:
        print(f"{sys.argv[0]} algo [digit]")
        exit(1)

    assert algo.lower() in [
        "svmlin",
    ]

    eval_algo(algo)


if __name__ == "__main__":

    main()
