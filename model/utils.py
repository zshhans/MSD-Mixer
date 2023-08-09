import os
import torch.nn as nn
from pytorch_lightning.loggers import CSVLogger


def get_csv_logger(save_dir, name):
    root_dir = os.fspath(os.path.join(save_dir, name))
    if not os.path.isdir(root_dir):
        return CSVLogger(save_dir, name)
    else:
        existing_versions = []
        for d in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir,
                                          d)) and d.startswith("version_"):
                existing_versions.append(int(d.split("_")[1]))

        if len(existing_versions) == 0:
            return CSVLogger(save_dir, name)

        return CSVLogger(save_dir, name, max(existing_versions) + 1)


def get_activation(activ: str):
    if activ == "gelu":
        return nn.GELU()
    elif activ == "sigmoid":
        return nn.Sigmoid()
    elif activ == "tanh":
        return nn.Tanh()
    elif activ == "relu":
        return nn.ReLU()

    raise RuntimeError("activation should not be {}".format(activ))


def get_loss_fn(loss_fn: str):
    if loss_fn == "mse":
        return nn.MSELoss()
    elif loss_fn == "mae":
        return nn.L1Loss()
    elif loss_fn == "huber":
        return nn.HuberLoss(delta=.1)

    raise RuntimeError("loss function should not be {}".format(loss_fn))
