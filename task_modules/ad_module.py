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
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from utils.tools import adjustment
from sklearn.metrics import precision_recall_fscore_support


class ADModule(pl.LightningModule):

    def __init__(self, config) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.out_chn = config.out_chn
        self.lambda_mse = config.lambda_mse
        self.lambda_acf = config.lambda_acf
        self.acf_cutoff = config.acf_cutoff
        self.optim = config.optim
        self.lr = config.lr
        self.lr_factor = config.lr_factor
        self.weight_decay = config.weight_decay
        self.patience = config.patience
        self.anomaly_ratio = config.anomaly_ratio
        self.model = MSDMixer(config.seq_len, config.pred_len, config.in_chn,
                              config.ex_chn, config.out_chn,
                              config.patch_sizes, config.hid_len,
                              config.hid_chn, config.hid_pch, config.hid_pred,
                              config.norm, config.use_last_norm, config.activ,
                              config.drop)

        self.val_mse = MeanSquaredError()
        self.val_mae = MeanAbsoluteError()
        self.test_mse = MeanSquaredError()
        self.test_mae = MeanAbsoluteError()
        self.final_mse = MeanSquaredError()
        self.final_mae = MeanAbsoluteError()
        self.loss_fn = get_loss_fn(config.loss_fn)
        self.anomaly_criterion = nn.MSELoss(reduce=False)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.float()
        _, res = self.model(x)
        residual_loss = residual_loss_fn(res, self.lambda_mse, self.lambda_acf,
                                       self.acf_cutoff)

        return residual_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        x = x.float()
        _, res = self.model(x)
        y_pred = x - res.transpose(1, 2)
        self.val_mse(y_pred, x)
        self.log("val_mse", self.val_mse)
        self.log("lr", self.optimizers().param_groups[0]['lr'])


    def on_test_epoch_start(self) -> None:
        self.train_energy = []
        self.test_energy = []
        self.test_labels = []
        return

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        x = x.float()
        _, res = self.model(x)
        y_pred = x - res.transpose(1, 2)
        score = torch.mean(self.anomaly_criterion(y_pred, x),
                           dim=-1).detach().cpu().numpy()
        if dataloader_idx == 0:
            self.train_energy.append(score)
        elif dataloader_idx == 1:
            self.test_energy.append(score)
            self.test_labels.append(y.cpu().numpy())

    def on_test_epoch_end(self) -> None:
        train_energy = np.concatenate(self.train_energy, axis=0).reshape(-1)
        test_energy = np.concatenate(self.test_energy, axis=0).reshape(-1)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold = np.percentile(combined_energy, 100 - self.anomaly_ratio)
        tqdm.write(f"Threshold: {threshold}")
        pred = (test_energy > threshold).astype(int)
        test_labels = np.concatenate(self.test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)
        gt, pred = adjustment(gt, pred)
        # accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, _ = precision_recall_fscore_support(
            gt, pred, average='binary')
        self.log("test_precision", precision)
        self.log("test_recall", recall)
        self.log("test_f_score", f_score)

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
            "min",
            factor=self.lr_factor,
            patience=self.patience,
            verbose=False,
            min_lr=1e-7)
        scheduler_config["monitor"] = "val_mse"
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
        }
