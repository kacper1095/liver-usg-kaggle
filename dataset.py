from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import imutils
import cv2
import imgaug.augmenters as iaa
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from skimage import exposure
from torch.utils.data import Dataset


class NormalizeHist:
    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = exposure.equalize_hist(img).astype(np.uint8)
        return img


class ElasticTransform:
    def __init__(self,
                 alpha: float,
                 sigma: float,
                 random_state: Optional[int] = None):
        self.random_state = random_state
        self.alpha = alpha
        self.sigma = sigma
        self.rng = np.random.RandomState(self.random_state)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        shape = img.shape
        dx = gaussian_filter((self.rng.rand(*shape) * 2 - 1),
                             self.sigma, mode="constant", cval=0) * self.alpha
        dy = gaussian_filter((self.rng.rand(*shape) * 2 - 1),
                             self.sigma, mode="constant", cval=0) * self.alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        distored_image = map_coordinates(img, indices, order=1, mode='reflect')
        return distored_image.reshape(img.shape)


class Denoising:
    def __init__(self, denoising_scale: int):
        self.denoising_scale = denoising_scale

    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = cv2.fastNlMeansDenoising(img, h=self.denoising_scale)
        return img


class ImgaugWrapper:
    def __init__(self, augmenter: iaa.Augmenter):
        self.augmenter = augmenter

    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = self.augmenter.augment_image(img)
        return img


class ToBGR:
    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img


class UsgDataset(Dataset):
    def __init__(self,
                 paths: Union[List[Path], Tuple[Path]],
                 is_train_or_valid: bool,
                 transforms: Optional[Callable] = None):
        self.paths = paths
        self.is_train_or_valid = is_train_or_valid
        self.transforms = transforms

    def __getitem__(self, index):
        a_path = self.paths[index]
        names = ["lower.png", "radial_polar_area.png", "circle.png"]
        stack = []
        for name in names:
            img = cv2.imread((a_path / name).as_posix(), cv2.IMREAD_GRAYSCALE)
            if name in ["lower.png", "circle.png"]:
                img = imutils.resize(img, height=128, inter=cv2.INTER_LANCZOS4)

            if self.transforms is not None:
                img = self.transforms(img)

            stack.append(img)

        img = np.concatenate(stack, axis=0)
        if self.is_train_or_valid:
            a_class = int(a_path.parent.name)
            return img, a_class

        return img, -1

    def __len__(self):
        return len(self.paths)
