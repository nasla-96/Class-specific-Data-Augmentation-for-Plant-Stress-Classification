import random
from typing import Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from scipy import ndimage


def transform_matrix_offset_center(matrix: np.ndarray, x: int, y: int) -> np.ndarray:
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    return offset_matrix @ matrix @ reset_matrix


def _apply_affine(img: Image.Image, transform_matrix: np.ndarray) -> Image.Image:
    arr = np.array(img)
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    transformed = np.stack(
        [
            ndimage.affine_transform(arr[:, :, c], affine_matrix, offset)
            for c in range(arr.shape[2])
        ],
        axis=2,
    )
    return Image.fromarray(transformed.astype(np.uint8))


def shear_x(img: Image.Image, magnitude: int = 8) -> Image.Image:
    arr = np.array(img)
    magnitudes = np.linspace(-0.3, 0.3, 11)
    transform_matrix = np.array(
        [[1, random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]), 0], [0, 1, 0], [0, 0, 1]]
    )
    transform_matrix = transform_matrix_offset_center(transform_matrix, arr.shape[0], arr.shape[1])
    return _apply_affine(img, transform_matrix)


def shear_y(img: Image.Image, magnitude: int = 7) -> Image.Image:
    arr = np.array(img)
    magnitudes = np.linspace(-0.3, 0.3, 11)
    transform_matrix = np.array(
        [[1, 0, 0], [random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]), 1, 0], [0, 0, 1]]
    )
    transform_matrix = transform_matrix_offset_center(transform_matrix, arr.shape[0], arr.shape[1])
    return _apply_affine(img, transform_matrix)


def translate_x(img: Image.Image, magnitude: int = 9) -> Image.Image:
    arr = np.array(img)
    magnitudes = np.linspace(-150 / 331, 150 / 331, 11)
    transform_matrix = np.array(
        [[1, 0, 0], [0, 1, arr.shape[1] * random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1])], [0, 0, 1]]
    )
    transform_matrix = transform_matrix_offset_center(transform_matrix, arr.shape[0], arr.shape[1])
    return _apply_affine(img, transform_matrix)


def translate_y(img: Image.Image, magnitude: int = 9) -> Image.Image:
    arr = np.array(img)
    magnitudes = np.linspace(-150 / 331, 150 / 331, 11)
    transform_matrix = np.array(
        [[1, 0, arr.shape[0] * random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1])], [0, 1, 0], [0, 0, 1]]
    )
    transform_matrix = transform_matrix_offset_center(transform_matrix, arr.shape[0], arr.shape[1])
    return _apply_affine(img, transform_matrix)


def rotate(img: Image.Image, magnitude: int = 3) -> Image.Image:
    arr = np.array(img)
    magnitudes = np.linspace(-30, 30, 11)
    theta = np.deg2rad(random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]))
    transform_matrix = np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    )
    transform_matrix = transform_matrix_offset_center(transform_matrix, arr.shape[0], arr.shape[1])
    return _apply_affine(img, transform_matrix)


def auto_contrast(img: Image.Image, magnitude: int = 8) -> Image.Image:
    return ImageOps.autocontrast(img)


def invert(img: Image.Image, magnitude: int = 3) -> Image.Image:
    return ImageOps.invert(img)


def equalize(img: Image.Image, magnitude: int = 5) -> Image.Image:
    return ImageOps.equalize(img)


def solarize(img: Image.Image, magnitude: int = 8) -> Image.Image:
    magnitudes = np.linspace(0, 256, 11)
    return ImageOps.solarize(img, random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]))


def posterize(img: Image.Image, magnitude: int = 7) -> Image.Image:
    magnitudes = np.linspace(4, 8, 11)
    bits = int(round(random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1])))
    return ImageOps.posterize(img, bits)


def contrast(img: Image.Image, magnitude: int = 7) -> Image.Image:
    magnitudes = np.linspace(0.1, 1.9, 11)
    return ImageEnhance.Contrast(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]))


def color(img: Image.Image, magnitude: int = 7) -> Image.Image:
    magnitudes = np.linspace(0.1, 1.9, 11)
    return ImageEnhance.Color(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]))


def brightness(img: Image.Image, magnitude: int = 6) -> Image.Image:
    magnitudes = np.linspace(0.1, 1.9, 11)
    return ImageEnhance.Brightness(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]))


def sharpness(img: Image.Image, magnitude: int = 6) -> Image.Image:
    magnitudes = np.linspace(0.1, 1.9, 11)
    return ImageEnhance.Sharpness(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]))


def cutout(img: Image.Image, magnitude: Optional[int] = None) -> Image.Image:
    arr = np.array(img).copy()
    magnitudes = np.linspace(0, 60 / 331, 11)
    mask_val = arr.mean()
    if magnitude is None:
        mask_size = 64
    else:
        mask_size = int(round(arr.shape[0] * random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1])))
    top = np.random.randint(0 - mask_size // 2, arr.shape[0] - mask_size)
    left = np.random.randint(0 - mask_size // 2, arr.shape[1] - mask_size)
    bottom = top + mask_size
    right = left + mask_size
    top = max(0, top)
    left = max(0, left)
    arr[top:bottom, left:right, :].fill(mask_val)
    return Image.fromarray(arr.astype(np.uint8))


AUGMENTATION_FUNCTIONS = {
    "shear_x": shear_x,
    "shear_y": shear_y,
    "translate_x": translate_x,
    "translate_y": translate_y,
    "rotate": rotate,
    "auto_contrast": auto_contrast,
    "invert": invert,
    "equalize": equalize,
    "solarize": solarize,
    "posterize": posterize,
    "contrast": contrast,
    "color": color,
    "brightness": brightness,
    "sharpness": sharpness,
    "cutout": cutout,
}
