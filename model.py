import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from torch.utils.data import DataLoader

import torchvision.transforms as transforms

from mnist_dataloader import MNIST_blur

import matplotlib.pyplot as plt

from dctutil import filter_deblur, err, gen_tikhonov_filters, gen_tsvd_filters

from scipy.fft import idctn

import numpy as np

import argparse


random.seed(3141592)


def loss_fn(x, x_hat):
    return F.l1_loss(x, x_hat)


class BBFFNN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, 128)
        self.l2 = nn.Linear(28 * 28, 128)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, 28 * 28)

        self.loss = loss_fn

    def forward(self, img_dct, psf_eig):
        z1 = self.l1(img_dct.view(-1, 28 * 28))
        z2 = self.l2(psf_eig.view(-1, 28 * 28))
        z3 = F.relu(torch.cat((z1, z2), dim=1))
        z4 = F.relu(self.l3(z3))
        s = (self.l4(z4)).view(-1, 1, 28, 28)
        return s

    def training_step(self, batch, batch_idx):
        blur_dct, S, true_filt = batch
        filt = self.forward(blur_dct, S)
        loss = self.loss(filt, true_filt)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        blur_dct, S, true_filt = batch
        filt = self.forward(blur_dct, S)
        loss = self.loss(filt, true_filt)
        self.log("test_loss", loss)

        true_deblur, learned_deblur = reconstruct(blur_dct, S, true_filt, filt)
        self.log("reconstruction_err", err(true_deblur, learned_deblur))

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def train():
    data_train = MNIST_blur(
        './datasets/MNIST',
        train=True,
        download=True,
        transform=transforms.ToTensor())

    data_test = MNIST_blur(
        './datasets/MNIST',
        train=False,
        download=True,
        transform=transforms.ToTensor())

    dl_train = DataLoader(data_train, batch_size=20, num_workers=8)
    dl_test = DataLoader(data_test, batch_size=20, num_workers=8)

    model = BBFFNN()
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=2
    )

    trainer.fit(
        model=model,
        train_dataloaders=dl_train,
        val_dataloaders=dl_test)


def reconstruct(blur_dct_t, S_t, true_filt_t, learned_filt_t):
    blur_dct = blur_dct_t.cpu().numpy()[0]

    S = S_t.cpu().numpy()[0]
    true_filt = true_filt_t.cpu().numpy()[0]
    learned_filt = learned_filt_t.cpu().detach().numpy()[0][0]

    true_deblur = filter_deblur(blur_dct, S, true_filt)
    learned_deblur = filter_deblur(blur_dct, S, learned_filt)

    return true_deblur, learned_deblur


def infer():
    model = BBFFNN.load_from_checkpoint(
        "lightning_logs/version_20/checkpoints/epoch=1-step=6000.ckpt")

    data = MNIST_blur(
        './datasets/MNIST',
        train=False,
        download=True,
        transform=transforms.ToTensor())

    blur_dct_t, S_t, true_filt_t = data[random.randint(0, 1000)]
    print(blur_dct_t.shape)

    blur_dct = blur_dct_t.numpy()[0]
    blurred = idctn(blur_dct)

    S = S_t.numpy()[0]
    true_filt = true_filt_t.numpy()[0]
    learned_filt_t = model.forward(blur_dct_t, S_t)
    learned_filt = learned_filt_t.detach().numpy()[0][0]

    true_deblur = filter_deblur(blur_dct, S, true_filt)
    learned_deblur = filter_deblur(blur_dct, S, learned_filt)

    tikhonov_filts = gen_tikhonov_filters(S, 0.1)
    tsvd_filts = gen_tsvd_filters(S, 12)

    tikhonov = filter_deblur(blur_dct, S, tikhonov_filts)
    tsvd = filter_deblur(blur_dct, S, tsvd_filts)

    plt.subplot(341)
    plt.title("true deblur")
    plt.imshow(true_deblur)

    plt.subplot(342)
    plt.title("blur dct")
    plt.imshow(blur_dct)

    plt.subplot(343)
    plt.title("blurred")
    plt.imshow(blurred)

    plt.subplot(344)
    plt.title("true filter factors")
    a = np.where(true_filt < -2, -2, true_filt)
    a = np.where(a > 2, 2, a)
    plt.imshow(a)

    plt.subplot(345)
    plt.title("learned filter factors")
    plt.imshow(learned_filt)

    plt.subplot(346)
    plt.title("learned deblur")
    plt.imshow(learned_deblur)

    plt.subplot(347)
    plt.title("tikhonov")
    plt.imshow(tikhonov)

    plt.subplot(348)
    plt.title("tsvd")
    plt.imshow(tsvd)

    plt.subplot(349)
    plt.title("tikhonov FF")
    plt.imshow(tikhonov_filts)
    
    plt.subplot(3, 4, 10)
    plt.title("tsvd FF")
    plt.imshow(tsvd_filts)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    if args.train:
        train()
    else:
        infer()


if __name__ == '__main__':
    main()
