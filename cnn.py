import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

import pandas as pd


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 4, 3)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(4, 8, 5)
        self.fc = nn.Linear(8 * 9 * 9, 10)

    def forward(self, x):
        z1 = F.relu(self.conv1(x))
        z2 = self.pool(z1)
        z3 = F.relu(self.conv2(z2))
        s = self.fc(z3.view(-1, 8 * 9 * 9))
        return s


def main():
    MNIST_train = datasets.MNIST(
        './datasets/MNIST',
        train=True,
        download=True,
        transform=transforms.ToTensor())

    MNIST_test = datasets.MNIST(
        './datasets/MNIST',
        train=False,
        download=True,
        transform=transforms.ToTensor())

    #  cnn = CNN()

    config = {
        'epochs': 5,
        'lr': 1e-3,
        'batches': 20
    }

    train_loader = DataLoader(
        MNIST_train,
        shuffle=True,
        batch_size=config['batches'])

    test_loader = DataLoader(
        MNIST_test,
        shuffle=True,
        batch_size=config['batches'])

    n_train = len(MNIST_train)
    n_test = len(MNIST_test)
    n_mb_train = len(train_loader)
    n_mb_test = len(test_loader)

    cnn = CNN()
    print(cnn)

    optimizer = torch.optim.Adam(cnn.parameters(), lr=config['lr'])

    ce_loss = nn.CrossEntropyLoss()

    df = pd.DataFrame(
        index=range(config['epochs']),
        columns=('epoch',
                 'loss_train',
                 'loss_test',
                 'acc_train',
                 'acc_test'))

    for epoch in range(config['epochs']):
        rl_train = 0
        acc_train = 0

        cnn.train()

        for X, y in train_loader:
            optimizer.zero_grad()
            score = cnn(X)
            loss = ce_loss(score, y)
            loss.backward()
            optimizer.step()

            rl_train += loss.detach().numpy()
            acc_train += (score.argmax(dim=1) == y).sum().numpy()

        rl_test = 0
        acc_test = 0
        with torch.no_grad():
            for X, y in test_loader:
                score = cnn(X)
                loss = ce_loss(score, y)
                rl_test += loss.detach().numpy()
                acc_test += (score.argmax(dim=1) == y).sum().numpy()

        loss_train = rl_train / n_mb_train
        loss_test = rl_test / n_mb_test

        acc_train /= n_train
        acc_test /= n_test

        print(epoch, loss_train, loss_test, acc_train, acc_test)
        df.loc[epoch] = [epoch, loss_train, loss_test, acc_train, acc_test]


if __name__ == '__main__':
    main()
