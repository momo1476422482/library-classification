import torch
from torch.utils.data import DataLoader
from dataset import MixGaussianGenerator
from dataset import MixGaussianDataset
import sys
from torch import nn
from model import MLP
from train import BaseTrainer
import numpy as np
import matplotlib.pyplot as plt


def eval_algo(algo: str) -> None:

    # parameters
    data_dim = 2
    data_size = 200
    num_epochs = 100

    # preparation of data
    data_generator = MixGaussianGenerator(
        (np.array([0, 0.5]), np.array([5, 5])),
        (0.5, 0.6),
        data_dimension=data_dim,
        flag_plot=True,
        data_size=data_size,
    )

    train_set = MixGaussianDataset(data_generator, train=True)
    train_loader = DataLoader(train_set, batch_size=20)

    test_set = MixGaussianDataset(data_generator, train=False)
    test_loader = DataLoader(test_set, batch_size=len(test_set))

    # Model construction

    if algo.lower() == "linear":
        model = MLP(data_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

    print(f"run {algo.upper()}")

    # train phase
    trainer = BaseTrainer(model, optimizer, nn.CrossEntropyLoss(), num_epochs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(num_epochs):
        trainer.train(epoch, train_loader)
        for it, (x, y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)
            score_test = model(x.float()).cpu().detach().numpy()
            prediction = np.argmax(score_test, axis=1).reshape(-1, 1)
            accuracy = np.mean((prediction == y.cpu().detach().numpy()))

    # print the accuracy of classification model
    print(f"accuracy is :{accuracy}")

    # plot the trained model
    w = model.state_dict()["layer.weight"].cpu().detach().numpy().tolist()
    w = w[0]
    print(w)
    b = model.state_dict()["layer.bias"].cpu().detach().numpy()
    print(b)
    a = -w[0] / w[1]
    xx = np.linspace(-2, 6)
    yy = a * xx - b / w[1]
    plt.plot(xx, yy, "k-", label="non weighted div")
    plt.savefig("model.png")


def main():

    if len(sys.argv) == 2:
        algo = sys.argv[1]
    else:
        print("no algorithme is called")
        exit(1)

    assert algo.lower() in ["linear"]

    eval_algo(algo)


if __name__ == "__main__":

    main()
