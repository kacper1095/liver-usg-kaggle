from pathlib import Path
from typing import Callable, List, Mapping, Optional, Tuple, Union

import cv2
import numpy as np
from skimage import exposure
from torch.utils.data import Dataset


class NormalizeHist:
    def __call__(self, img: np.ndarray):
        img = img.transpose((2, 0, 1))
        img[0] = exposure.equalize_hist(img[0]).astype(np.uint8)
        img[1] = exposure.equalize_hist(img[1]).astype(np.uint8)
        img[2] = exposure.equalize_hist(img[2]).astype(np.uint8)
        img = img.transpose((1, 2, 0))
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
        img = cv2.imread(a_path.as_posix())

        if self.transforms is not None:
            img = self.transforms(img)

        if self.is_train_or_valid:
            a_class = int(a_path.parent.parent.name)
            return img, a_class

        return img, -1

    def __len__(self):
        return len(self.paths)
