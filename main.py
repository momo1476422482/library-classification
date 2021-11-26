import torch
from torch.utils.data import DataLoader
from dataset import MixGaussianGenerator
from dataset import MixGaussianDataset
from model import ModelSVM
import sys
from torch import nn
from model import MLP
from train import BaseTrainer
import numpy as np


def eval_algo(algo: str) -> None:

    data_generator = MixGaussianGenerator((np.array([0, 1]), np.array([5, 5])), (0.5, 0.6))
    num_epochs=50

    train_set = MixGaussianDataset(data_generator,train=True)
    train_loader = DataLoader(train_set, batch_size=20)

    test_set = MixGaussianDataset(data_generator, train=False)
    test_loader = DataLoader(test_set, batch_size=len(test_set))

    if algo.lower() == "svmlin":

        model = ModelSVM()

    elif algo.lower() == "linear":
        model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

    print(f"run {algo.upper()}")

    # train phase
    trainer=BaseTrainer(model,optimizer,nn.CrossEntropyLoss(),num_epochs)

    for epoch in range(num_epochs):
        trainer.train(epoch,train_loader)
        for it,batch in enumerate(test_loader):
            batch[0] = batch[0].to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            score_test = model(batch[0].float()).cpu().detach().numpy()
            prediction = np.argmax(score_test, axis=1).reshape(-1,1)
            accuracy = np.mean((prediction == batch[1].cpu().detach().numpy()))

    # print the accuracy of classification model
    print("accuracy is", accuracy)


def main():

    if len(sys.argv) == 2:
        algo = sys.argv[1]
    else:
        print(f"{sys.argv[0]} algo [digit]")
        exit(1)

    assert algo.lower() in ["svmlin", "linear"]

    eval_algo(algo)


if __name__ == "__main__":

    main()
