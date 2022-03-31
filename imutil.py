from enum import Enum
from PIL import Image
import numpy as np


def psf_moffat(dim, s, beta):
    x = np.arange(0, dim[0])
    y = np.arange(0, dim[1])
    X, Y = np.meshgrid(x, y)
    inv_c = s ** -2
    x_shift = dim[0] // 2
    y_shift = dim[1] // 2
    matrix_product = inv_c * ((X - x_shift) ** 2 + (Y - y_shift) ** 2)
    return (1 + matrix_product) ** -beta


def psf_gauss(dim, s, beta=0):
    x = np.arange(0, dim[0])
    y = np.arange(0, dim[1])
    X, Y = np.meshgrid(x, y)
    inv_c = s ** -2
    x_shift = dim[0] // 2
    y_shift = dim[1] // 2
    matrix_product = inv_c * ((X - x_shift) ** 2 + (Y - y_shift) ** 2)
    return np.exp(-0.5 * matrix_product)


def get_img(path):
    pil_image = Image.open(path)
    dtype = {'F': np.float32, 'L': np.uint8, 'RGB': np.uint8}[pil_image.mode]
    np_image = np.array(pil_image.getdata(), dtype=dtype)
    w, h = pil_image.size
    np_image.shape = (h, w)
    return np_image


class BCType(Enum):
    ZERO = 0
    PERIODIC = 1
    REFLEXIVE = 2


def __gen_zero_bc(img):
    h, w = img.shape
    zero_bc_ud = np.zeros([h, w * 3])
    zero_bc = np.zeros([h, w])
    c_stack = np.hstack([zero_bc, img, zero_bc])
    X_ext = np.vstack([zero_bc_ud, c_stack, zero_bc_ud])
    return X_ext


def __gen_periodic_bc(img):
    c_stack = np.hstack([img, img, img])
    X_ext = np.vstack([c_stack, c_stack, c_stack])
    return X_ext


def __gen_reflexive_bc(img):
    lr = np.fliplr(img)
    ud = np.flipud(img)
    both = np.flipud(lr)

    t_stack = np.hstack([both, ud, both])
    c_stack = np.hstack([lr, img, lr])
    X_ext = np.vstack([t_stack, c_stack, t_stack])
    return X_ext


def gen_bc(img, bc_type: BCType):
    if bc_type == BCType.ZERO:
        return __gen_zero_bc(img)
    elif bc_type == BCType.PERIODIC:
        return __gen_periodic_bc(img)
    elif bc_type == BCType.REFLEXIVE:
        return __gen_reflexive_bc(img)

