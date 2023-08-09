import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import *
import pytorch_lightning as pl
from model.residual_loss import residual_loss_fn
from model.utils import get_loss_fn
from model.msd_mixer import MSDMixer
from tqdm import tqdm
import numpy as np
from torchmetrics import Accuracy


class CLSModule(pl.LightningModule):

    def __init__(self, in_len, out_len, in_chn, out_chn, ex_chn, patch_sizes,
                 hid_len, hid_chn, hid_pch, hid_pred, norm, use_last_norm,
                 drop, lambda_mse, lambda_acf, acf_cutoff, optim, lr,
                 lr_factor, weight_decay, patience, activ, reduction,
                 batch_size) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.out_chn = out_chn
        self.lambda_mse = lambda_mse
        self.lambda_acf = lambda_acf
        self.acf_cutoff = acf_cutoff
        self.optim = optim
        self.lr = lr
        self.lr_factor = lr_factor
        self.weight_decay = weight_decay
        self.patience = patience
        self.batch_size = batch_size
        self.model = MSDMixer(in_len, out_len, in_chn, ex_chn, out_chn,
                              patch_sizes, hid_len, hid_chn, hid_pch, hid_pred,
                              norm, use_last_norm, activ, drop, reduction)

        self.train_acc = Accuracy("multiclass", num_classes=out_chn)
        self.val_acc = Accuracy("multiclass", num_classes=out_chn)
        self.test_acc = Accuracy("multiclass", num_classes=out_chn)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        x = x.float()
        y = y.squeeze(-1).long()
        y_pred, res = self.model(x, None)
        y_pred = y_pred.squeeze(1)
        self.train_acc(y_pred, y)
        pred_loss = F.cross_entropy(y_pred, y)
        residual_loss = residual_loss_fn(res, self.lambda_mse, self.lambda_acf,
                                        self.acf_cutoff)
        # self.log("pred_loss", pred_loss, on_step=False, on_epoch=True)
        # self.log("residual_loss", residual_loss, on_step=False, on_epoch=True)
        loss = pred_loss + residual_loss

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        x = x.float()
        y = y.squeeze(-1).long()
        y_pred, _ = self.model(x, None)
        y_pred = y_pred.squeeze(1)
        pred_loss = F.cross_entropy(y_pred, y)
        self.val_acc(y_pred, y)
        self.log("val_acc", self.val_acc)
        self.log("val_loss", pred_loss)
        self.log("lr", self.optimizers().param_groups[0]['lr'])
        return

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        x = x.float()
        y = y.squeeze(-1).long()
        y_pred, _ = self.model(x, None)
        y_pred = y_pred.squeeze(1)
        self.test_acc(y_pred, y)
        self.log("test_acc", self.test_acc)
        return

    def configure_optimizers(self):
        if self.optim == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(),
                                          lr=self.lr,
                                          weight_decay=self.weight_decay)
        elif self.optim == "adam":
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.lr,
                                         weight_decay=self.weight_decay)
        else:
            raise ValueError
        scheduler_config = {}
        # scheduler_config["scheduler"] = ExponentialLR(optimizer,0.5)
        scheduler_config["scheduler"] = ReduceLROnPlateau(
            optimizer,
            'min',
            factor=self.lr_factor,
            patience=self.patience,
            verbose=True,
            min_lr=1e-7)
        scheduler_config["monitor"] = "val_loss"
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
        }
