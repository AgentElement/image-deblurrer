import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from torchvision import datasets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from scipy.fft import dctn, idctn
import pandas as pd

from dctutil import noisy_blur, pad_to_image, eigenvalues, filter_deblur, err
from imutil import psf_gauss


random.seed(3141592)

config = {
    'epochs': 5,
    'lr': 1e-3,
    'batches': 20
}


class BBFFNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, 128)
        self.l2 = nn.Linear(28 * 28, 128)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, 28 * 28)

    def forward(self, img_dct, psf_eig):
        z1 = self.l1(img_dct.view(-1, 28 * 28))
        z2 = self.l2(psf_eig.view(-1, 28 * 28))
        z3 = F.relu(torch.cat((z1, z2), dim=1))
        z4 = F.relu(self.l3(z3))
        s = (self.l4(z4)).view(-1, 1, 28, 28)
        return s


def load_mnist_dl(batch_size):
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

    train_loader = DataLoader(
        MNIST_train,
        shuffle=True,
        batch_size=batch_size)

    test_loader = DataLoader(
        MNIST_test,
        shuffle=True,
        batch_size=batch_size)

    data_config = {
        "n_train": len(MNIST_train),
        "n_test": len(MNIST_test),
        "n_mb_train": len(train_loader),
        "n_mb_test": len(test_loader),
    }

    return MNIST_test, MNIST_train, train_loader, test_loader, data_config


def generate_problem(img):
    psf_factor = random.uniform(0.5, 2)
    noise_factor = random.uniform(0, 0.05)

    P = psf_gauss(img.shape, psf_factor)
    blurred_image = noisy_blur(img, P, noise_factor)

    S = eigenvalues(P)

    blur_dct = dctn(blurred_image)
    true_dct = dctn(img)

    true_filt = S * true_dct / blur_dct

    return blur_dct, S, true_filt


def convert_batch_to_problem(batch):
    blur_dct_batch = torch.empty(size=batch.size())
    eig_batch = torch.empty(size=batch.size())
    true_filt_batch = torch.empty(size=batch.size())

    for i, b in enumerate(batch):
        for j, channel in enumerate(b):
            blur_dct, eig, true_filt = generate_problem(channel.numpy())
            blur_dct_batch[i][j] = torch.tensor(blur_dct)
            eig_batch[i][j] = torch.tensor(eig)
            true_filt_batch[i][j] = torch.tensor(true_filt)

    return blur_dct_batch, eig_batch, true_filt_batch


def main():
    _, _, train_loader, test_loader, data_config = load_mnist_dl(
        config['batches'])

    model = BBFFNN()

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    mse_loss = nn.MSELoss()

    df = pd.DataFrame(
        index=range(config['epochs']),
        columns=('epoch',
                 'loss_train',
                 'loss_test',
                 ))

    for epoch in range(config['epochs']):
        rl_train = 0

        model.train()

        for X, y in train_loader:
            blur_dct, S, true_filt = convert_batch_to_problem(X)

            optimizer.zero_grad()
            score = model(blur_dct, S)
            loss = mse_loss(score, true_filt)
            loss.backward()
            optimizer.step()

            rl_train += loss.detach().numpy()

        rl_test = 0
        with torch.no_grad():
            for X, y in test_loader:
                blur_dct, S, true_filt = convert_batch_to_problem(X)

                score = model(blur_dct, S)
                loss = mse_loss(score, true_filt)
                rl_test += loss.detach().numpy()

        loss_train = rl_train / data_config['n_mb_train']
        loss_test = rl_test / data_config['n_mb_test']

        torch.save(model.state_dict(), f"models{epoch}.pt")
        print(epoch, loss_train, loss_test)
        df.loc[epoch] = [epoch, loss_train, loss_test]


def test():
    test, train, train_loader, test_loader, data_config = load_mnist_dl(
        config['batches'])

    PSF = psf_gauss((28, 28), 1, 1)
    X, y = test[0]
    true_image = X[0].numpy()

    P = pad_to_image(true_image, PSF)
    blurred_image = noisy_blur(X[0].numpy(), P, 0.01)

    S_unfilt = eigenvalues(P)

    blur_dct = dctn(blurred_image)
    true_dct = dctn(true_image)

    true_filt = S_unfilt * true_dct / blur_dct

    true_deblurred = filter_deblur(blur_dct, S_unfilt, true_filt)

    print(err(true_deblurred, true_image))

    plt.imshow(true_deblurred)
    plt.show()


def showmodel():
    model = BBFFNN()
    model.load_state_dict(torch.load("models2.pt"))
    model.eval()

    test, train, train_loader, test_loader, data_config = load_mnist_dl(
        config['batches'])

    interm, _ = test[0]

    blur_dct, S, true_filt = convert_batch_to_problem(test)
    print(blur_dct.shape)
    filt = model(blur_dct, S)
    deblurred = filter_deblur(blur_dct, S, filt)

    plt.subplot(221)
    plt.imshow(interm[0].numpy())
    plt.subplot(222)
    plt.imshow(deblurred)
    plt.show()


if __name__ == '__main__':
    showmodel()
