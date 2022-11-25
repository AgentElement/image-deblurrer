import numpy as np
from scipy.fft import dctn, idctn
from numpy.linalg import svd, norm

from imutil import AsymPSFError


def dctshift(PSF):
    x, y = PSF.shape
    # This is for correctness. The PSF is truncated and squared anyway
    # here, and that can be confusing. If the assertion is removed, dctshift
    # will work anyway
    if not x == y:
        raise AsymPSFError("PSF is not square")
    cx = x // 2
    cy = y // 2
    c = min(cx + x % 2, cy + y % 2, x - cx, y - cy)

    Z1 = np.diag(np.ones(c), c - 1)
    Z2 = np.diag(np.ones(c - 1), c)

    PP = PSF[cx - c + 1:cx + c, cy - c + 1:cy + c]
    PP = Z1 @ PP @ Z1.T \
        + Z1 @ PP @ Z2.T \
        + Z2 @ PP @ Z1.T \
        + Z2 @ PP @ Z2.T
    p, q = PP.shape
    return np.pad(PP, ((0, x - p), (0, y - p)))


def convolving_matrix(PSF):
    shift = dctshift(PSF)
    e1 = np.zeros(shift.shape)
    e1[0, 0] = 1
    return dctn(shift) / dctn(e1)


def pad_to_image(img, P):
    x, y = img.shape
    p, q = P.shape
    padded = np.zeros(img.shape)
    padded[(x - p) // 2:(x + p) // 2 + x % 2,
           (y - q) // 2:(y + q) // 2 + y % 2] = P
    return padded


def blur(img, PSF):
    P = pad_to_image(img, PSF)
    S = convolving_matrix(P)
    return idctn(S * dctn(img))


def noisy_blur(img, PSF, noise):
    blurred = blur(img, PSF)
    E = np.random.rand(*PSF.shape)
    E = E / norm(E)
    return blurred + noise * norm(blurred) * E


def naive_deblur(img, PSF):
    P = pad_to_image(img, PSF)
    S = convolving_matrix(P)
    return idctn(dctn(img) / S)


def filter_deblur(dct_img, eig, filter):
    return idctn(filter * dct_img / eig)


def trunc_svd(X, k):
    u, s, v = svd(X, full_matrices=False)
    return (u * np.append(s[:k], np.zeros(s.shape)[k:])) @ v


def err(X, S):
    return norm(X - S) / norm(X)


def gen_tikhonov_filters(S, alpha):
    return S**2 / (S**2 + alpha ** 2)


def gen_tsvd_filters(S, k):
    x, y = S.shape
    xm = np.arange(0, x)
    ym = np.arange(0, y)
    xx, yy = np.meshgrid(xm, ym)
    filt = xx ** 2 + yy ** 2
    return np.where(filt < k ** 2, 1, 0)
