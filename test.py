import numpy as np
from scipy.fft import dctn, idctn
from matplotlib import pyplot as plt
from numpy.linalg import svd, norm
from tqdm import tqdm

from imutil import psf_gauss, get_img

#  from skimage.measure import block_reduce


class AsymPSFError(Exception):
    pass


def psf_correct(PSF: np.ndarray):
    x, y = PSF.shape
    if not ((x % 2) and (y % 2)):
        raise AsymPSFError("PSF shape contains even components")


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


def eigenvalues(PSF):
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
    S = eigenvalues(P)
    return idctn(S * dctn(img))


def noisy_blur(img, PSF, noise):
    blurred = blur(img, PSF)
    E = np.random.rand(*PSF.shape)
    E = E / norm(E)
    return blurred + noise * norm(blurred) * E


def naive_deblur(img, PSF):
    P = pad_to_image(img, PSF)
    S = eigenvalues(P)
    return idctn(dctn(img) / S)


def trunc_svd(X, k):
    u, s, v = svd(X, full_matrices=False)
    return (u * np.append(s[:k], np.zeros(s.shape)[k:])) @ v


def err(X, S):
    return norm(X - S) / norm(X)


def compute_errors(blurred, PSF, original):
    errors = []
    u, s, v = svd(blurred, full_matrices=False)
    for i in range(1, 70):
        trunc = (u * np.append(s[:i], np.zeros(s.shape)[i:])) @ v
        deblurred = naive_deblur(blurred, trunc)
        error = err(original, deblurred)
        errors.append(error)
        print(error)
    plt.plot(range(1, 70), errors)


def main():
    img = get_img("Challenges/iogray.tif") / 255
    img.astype(np.float64)
    x, y = img.shape
    PSF = psf_gauss(img.shape, 10, 10)
    blurred = noisy_blur(img, PSF, 0.01)

    P = pad_to_image(img, PSF)
    spectral = 1
    S = eigenvalues(P)
    S_filt = np.where(S != 0, S, float('+inf'))
    blur_dct = dctn(blurred)


    # TRUNCATED SVD
    # Construct filter meshgrid centered at (0, 0)
    xm = np.arange(0, x)
    ym = np.arange(0, y)
    xx, yy = np.meshgrid(xm, ym)
    filt = xx ** 2 + yy ** 2

    tsvd_errors = []
    tsvd_deblurred = []
    for i in range(1, 70):
        trunc = i
        S_mask = np.where(filt < trunc ** 2, S_filt, float('+inf'))
        deblurred = idctn(blur_dct / S_mask)
        tsvd_deblurred.append(deblurred)
        error = err(img, deblurred)
        tsvd_errors.append(error)

    lowest_tsvd_error_index = np.argmin(tsvd_errors)
    print(tsvd_errors[lowest_tsvd_error_index])
    best_tsvd = tsvd_deblurred[lowest_tsvd_error_index]



    # TIKHONOV METHOD
    tikhonov_errors = []
    tikhonov_deblurred = []
    for i in np.logspace(0, 1, 70):
        alpha = i
        spectral = S ** 2 / (S ** 2 + alpha ** 2)
        deblurred = idctn(spectral * blur_dct / S_filt)
        tikhonov_deblurred.append(deblurred)
        error = err(img, deblurred)
        tikhonov_errors.append(error)

    lowest_tikhonov_error_index = np.argmin(tikhonov_errors)
    print(tikhonov_errors[lowest_tikhonov_error_index])
    best_tikhonov = tikhonov_deblurred[lowest_tikhonov_error_index]



    plt.subplot(321)
    plt.imshow(blurred)
    plt.subplot(322)
    plt.imshow(img)

    plt.subplot(323)
    plt.imshow(best_tsvd)
    plt.subplot(324)
    plt.imshow(best_tikhonov)
    
    plt.subplot(325)
    plt.plot(range(len(tsvd_errors)), np.log(tsvd_errors))
    plt.subplot(326)
    plt.plot(range(len(tikhonov_errors)), np.log(tikhonov_errors))
    plt.show()


def best_trunc(filt, blur_dct, spectral, img, deblurred, x, y, S_filt):
    errors = []
    for i in tqdm(range(min(x, y))):
        S_mask = np.where(filt < i ** 2, S_filt, float('+inf'))
        deblurred = idctn(spectral * blur_dct / S_mask)
        error = err(img, deblurred)
        errors.append(error)

    best_trunc = np.argmin(errors)
    return best_trunc


def reduce(image):
    pass


def test():
    PSF = psf_gauss((5, 5), 4, 4)
    img = np.random.rand(10, 10)

    P = pad_to_image(img, PSF)
    shifted = dctshift(P)
    plt.imshow(shifted)
    plt.show()


if __name__ == '__main__':
    main()
