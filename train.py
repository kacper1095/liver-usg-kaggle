import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNet
from skorch.callbacks import BatchScoring, Checkpoint, EpochScoring, LRScheduler, ProgressBar
from skorch.helper import predefined_split

from common import acc, fscore, fscore_as_metric, get_test_transformers, get_timestamp, get_train_test_split_from_paths, \
    get_train_valid_transformers
from dataset import UsgDataset
from model import PretrainedModel

# Needed it because of in `DataLoader` for validation set
# RuntimeError: received 0 items of ancdata
# https://github.com/pytorch/pytorch/issues/973#issuecomment-426559250
torch.multiprocessing.set_sharing_strategy('file_system')


def train(data_folder: str, out_model: str):
    out_model = Path(out_model)
    out_model.mkdir()

    data_paths = list((Path(data_folder) / "train").rglob("radial_polar_area.png"))
    data_paths = list(sorted(data_paths, key=lambda x: int(x.parent.name)))

    classes = [int(path.parent.parent.name) for path in data_paths]
    train_paths, valid_paths = get_train_test_split_from_paths(data_paths, classes)

    train_dataset = UsgDataset(train_paths, True, transforms=get_train_valid_transformers())
    valid_dataset = UsgDataset(valid_paths, True, transforms=get_train_valid_transformers())

    net = NeuralNet(
        PretrainedModel,
        criterion=nn.CrossEntropyLoss,
        batch_size=16,
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
        transforms=get_test_transformers())

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
