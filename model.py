import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from torch.utils.data import DataLoader

import torchvision.transforms as transforms

from mnist_dataloader import MNIST_blur

import matplotlib.pyplot as plt

from dctutil import filter_deblur

from scipy.fft import idctn

random.seed(3141592)


class BBFFNN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, 128)
        self.l2 = nn.Linear(28 * 28, 128)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, 28 * 28)

        self.loss = F.kl_div

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
        loss = F.mse_loss(filt, true_filt)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        blur_dct, S, true_filt = batch
        filt = self.forward(blur_dct, S)
        loss = F.mse_loss(filt, true_filt)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def main():
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
        max_epochs=10
    )

    trainer.fit(
        model=model,
        train_dataloaders=dl_train,
        val_dataloaders=dl_test)


def infer():
    model = BBFFNN.load_from_checkpoint(
        "lightning_logs/version_6/checkpoints/epoch=9-step=30000.ckpt")

    data = MNIST_blur(
        './datasets/MNIST',
        train=False,
        download=True,
        transform=transforms.ToTensor())

    blur_dct_t, S_t, true_filt_t = data[0]
    print(blur_dct_t.shape)

    blur_dct = blur_dct_t.numpy()[0]
    blurred = idctn(blur_dct)

    S = S_t.numpy()[0]
    true_filt = true_filt_t.numpy()[0]
    learned_filt_t = model.forward(blur_dct_t, S_t)
    learned_filt = learned_filt_t.detach().numpy()[0][0]

    true_deblur = filter_deblur(blur_dct, S, true_filt)
    learned_deblur = filter_deblur(blur_dct, S, learned_filt)

    plt.subplot(231)
    plt.title("true deblur")
    plt.imshow(true_deblur)

    plt.subplot(232)
    plt.title("blur dct")
    plt.imshow(blur_dct)

    plt.subplot(233)
    plt.title("blurred")
    plt.imshow(blurred)

    plt.subplot(234)
    plt.title("true filter factors")
    plt.imshow(true_filt)

    plt.subplot(235)
    plt.title("learned filter factors")
    plt.imshow(learned_filt)

    plt.subplot(236)
    plt.title("learned deblur")
    plt.imshow(learned_deblur)

    plt.show()


if __name__ == '__main__':
    #  main()
    infer()
