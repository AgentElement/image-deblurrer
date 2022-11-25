import numpy as np
from scipy.fft import dctn, idctn
from matplotlib import pyplot as plt
from numpy.linalg import svd
from tqdm import tqdm

from imutil import psf_gauss, get_img

#  from skimage.measure import block_reduce

from dctutil import naive_deblur, err, noisy_blur, pad_to_image, convolving_matrix, dctshift


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
    img = get_img("datasets/iogray.tif") / 255
    img.astype(np.float64)
    x, y = img.shape

    PSF = psf_gauss(img.shape, 10, 10)

    P = pad_to_image(img, PSF)
    S = convolving_matrix(P)
    S_filt = np.where(S != 0, S, float('+inf'))

    blurred = noisy_blur(img, PSF, 0.01)
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
        #  print(i, error)
        tsvd_errors.append(error)

    lowest_tsvd_error_index = np.argmin(tsvd_errors)
    print(tsvd_errors[lowest_tsvd_error_index])
    best_tsvd = tsvd_deblurred[lowest_tsvd_error_index]

    # TIKHONOV METHO
    tikhonov_errors = []
    tikhonov_deblurred = []
    for i in np.logspace(0, 1, 70):
        alpha = i
        spectral = S ** 2 / (S ** 2 + alpha ** 2)
        deblurred = idctn(spectral * blur_dct / S_filt)
        tikhonov_deblurred.append(deblurred)
        error = err(img, deblurred)
        print(alpha, error)
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


def show_psf():
    PSF = psf_gauss((5, 5), 4, 4)
    img = np.random.rand(10, 10)

    P = pad_to_image(img, PSF)
    shifted = dctshift(P)
    plt.imshow(shifted)
    plt.show()


def test():
    img = get_img("datasets/iogray.tif") / 255
    img.astype(np.float64)
    x, y = img.shape
    PSF = psf_gauss(img.shape, 10, 10)
    blurred = noisy_blur(img, PSF, 0.01)

    P = pad_to_image(img, PSF)
    alpha = 10
    S = convolving_matrix(P)
    spectral = S ** 2 / (S ** 2 + alpha ** 2)
    S_filt = np.where(S != 0, S, float('+inf'))

    blur_dct = dctn(blurred)
    true_dct = dctn(img)
    deblurred_dct = spectral / S_filt
    deblurred = idctn(spectral * blur_dct / S_filt)

    true = idctn(true_dct)

    plt.subplot(221)
    plt.imshow(np.log(np.abs(deblurred_dct)))
    plt.subplot(222)
    plt.imshow(deblurred)

    plt.subplot(223)
    plt.imshow(np.log(np.abs(true_dct / blur_dct)))
    plt.subplot(224)
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    main()
    #  test()
