import random

import numpy as np

import torch
from torch.utils.data import Dataset

from torchvision import datasets
import torchvision.transforms as transforms

from dctutil import convolving_matrix, noisy_blur, filter_deblur
from imutil import psf_gauss

from scipy.fft import dctn

import matplotlib.pyplot as plt


class MNIST_blur(Dataset):
    def __init__(self, file, train, download, transform):
        self.MNIST = datasets.MNIST(
            file,
            train=train,
            download=download,
            transform=transforms.ToTensor())
        self.transform = transform

    def __len__(self):
        return len(self.MNIST)

    def __getitem__(self, idx):
        img, _ = self.MNIST[idx]
        z, x, y = img.shape
        img = np.reshape(img.numpy(), (x, y))

        psf_factor = random.uniform(0.5, 1.5)
        noise_factor = random.uniform(0.01, 0.05)

        P = psf_gauss(img.shape, psf_factor)
        blurred_image = noisy_blur(img, P, noise_factor)
        S = convolving_matrix(P)

        blur_dct = dctn(blurred_image)
        true_dct = dctn(img)

        true_filt = S * true_dct / blur_dct

        return (self.transform(blur_dct.astype('float32')),
                self.transform(S.astype('float32')),
                self.transform(true_filt.astype('float32')))


def test_dataset():
    data = MNIST_blur(
        './datasets/MNIST',
        train=False,
        download=True,
        transform=transforms.ToTensor())

    blur_dct_t, S_t, true_filt_t = data[0]
    print(blur_dct_t.shape)

    blur_dct = blur_dct_t.numpy()[0]
    S = S_t.numpy()[0]
    true_filt = true_filt_t.numpy()[0]

    print(S.dtype)

    plt.subplot(221)
    true_deblur = filter_deblur(blur_dct, S, true_filt)
    plt.imshow(true_deblur)

    plt.subplot(222)
    plt.imshow(blur_dct)

    plt.subplot(223)
    plt.imshow(S)

    plt.subplot(224)
    plt.imshow(true_filt)
    plt.show()


if __name__ == "__main__":
    test_dataset()
