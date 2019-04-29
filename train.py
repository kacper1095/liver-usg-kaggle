import argparse
import datetime
from pathlib import Path
from typing import Optional

import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from skorch import NeuralNet
from skorch.callbacks import BatchScoring, Checkpoint, EpochScoring, LRScheduler, ProgressBar
from skorch.helper import predefined_split
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomCrop, Resize, ToPILImage, ToTensor

from dataset import Denoising, ToBGR, UsgDataset
from model import PretrainedModel

# Needed it because of in `DataLoader` for validation set
# RuntimeError: received 0 items of ancdata
# https://github.com/pytorch/pytorch/issues/973#issuecomment-426559250
torch.multiprocessing.set_sharing_strategy('file_system')


def acc_as_metric(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return (np.argmax(y_pred, axis=1) == y_true).mean().item()


def acc(net: NeuralNet,
        ds: Optional[Dataset] = None,
        y: Optional[torch.Tensor] = None,
        y_pred: Optional[torch.Tensor] = None) -> float:
    if y_pred is None:
        y_pred = net.predict(ds)
    return acc_as_metric(y_pred, y)


def fscore_as_metric(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return f1_score(y_true, np.argmax(y_pred, axis=1), average="weighted")


def fscore(net: NeuralNet,
           ds: Optional[Dataset] = None,
           y: Optional[torch.Tensor] = None,
           y_pred: Optional[torch.Tensor] = None) -> float:
    if y_pred is None:
        y_pred = net.predict(ds)
    return fscore_as_metric(y_pred, y)


def get_timestamp() -> str:
    return datetime.datetime.now().strftime("%HH_%MM_%dd_%mm_%Yy")


def train(data_folder: str, out_model: str):
    out_model = Path(out_model)
    out_model.mkdir()

    data_paths = list((Path(data_folder) / "train").rglob("radial_polar_area.png"))

    classes = [int(path.parent.parent.name) for path in data_paths]
    train_paths, valid_paths = train_test_split(data_paths, test_size=0.3, stratify=classes, random_state=0)

    train_augmenters = iaa.Sequential([
        iaa.Fliplr(p=0.2),
        iaa.Flipud(p=0.2),
        iaa.Affine(
            rotate=(-5, 5),
            translate_px=(-5, 5),
            mode=ia.ALL
        )
    ], random_order=True)

    transforms = Compose([
        Denoising(denoising_scale=7),
        # ImgaugWrapper(train_augmenters),
        ToBGR(),
        ToPILImage(),
        RandomCrop(128, pad_if_needed=True),
        Resize(128, interpolation=Image.LANCZOS),
        ToTensor()
    ])

    train_dataset = UsgDataset(train_paths, True, transforms=transforms)
    valid_dataset = UsgDataset(valid_paths, True, transforms=transforms)

    net = NeuralNet(
        PretrainedModel,
        criterion=nn.CrossEntropyLoss,
        batch_size=18,
        max_epochs=100,
        optimizer=optim.Adam,
        lr=0.0001,
        iterator_train__shuffle=True,
        iterator_train__num_workers=2,
        iterator_valid__shuffle=False,
        iterator_valid__num_workers=2,
        train_split=predefined_split(valid_dataset),
        device="cuda",
        callbacks=[
            Checkpoint(
                f_params=(out_model / "params.pt").as_posix(),
                f_optimizer=(out_model / "optim.pt").as_posix(),
                f_history=(out_model / "history.pt").as_posix()
            ),
            LRScheduler(
                policy="ReduceLROnPlateau",
                monitor="valid_loss",
                factor=0.25,
                patience=7,
            ),
            EpochScoring(acc, name="val_acc", lower_is_better=False, on_train=False),
            EpochScoring(fscore, name="val_fscore", lower_is_better=False, on_train=False),
            BatchScoring(acc, name="train_acc", lower_is_better=False, on_train=True),
            BatchScoring(fscore, name="train_fscore", lower_is_better=False, on_train=True),
            ProgressBar(postfix_keys=["train_loss", "train_fscore"]),
        ],
        warm_start=True
    )

    net.fit(train_dataset)

    net = NeuralNet(
        PretrainedModel,
        criterion=nn.CrossEntropyLoss,
        iterator_valid__shuffle=False,
        iterator_valid__num_workers=2,
        iterator_valid__batch_size=1,
        device="cuda",
    )
    net.initialize()
    net.load_params(f_params=(out_model / "params.pt").as_posix())

    test_data_paths = list((Path(data_folder) / "test").rglob("radial_polar_area.png"))
    test_dataset = UsgDataset(
        test_data_paths, is_train_or_valid=False,
        transforms=Compose([
            Denoising(denoising_scale=7),
            ToBGR(),
            ToTensor()
        ]))

    valid_predictions = net.predict(valid_dataset)
    valid_trues = np.asarray([int(path.parent.parent.name) for path in valid_paths])
    val_acc = fscore_as_metric(valid_predictions, valid_trues)

    predictions = net.predict(test_dataset)

    ids = [path.parent.name for path in test_data_paths]
    classes = np.argmax(predictions, axis=1)
    frame = pd.DataFrame(data={"id": ids, "label": classes})
    frame["id"] = frame["id"].astype(np.int)
    frame = frame.sort_values(by=["id"])
    frame.to_csv(f"submissions/{get_timestamp()}_{'%.4f' % val_acc}_submission.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_folder",
        help="Folder with 'train' and 'test' folders prepared for the competition."
    )
    parser.add_argument(
        "out_model",
        help="Output folder where weights and tensorboards will be saved."
    )

    args = parser.parse_args()
    train(args.data_folder, args.out_model)


if __name__ == '__main__':
    main()
