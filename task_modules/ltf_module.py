import torch
from torch.optim.lr_scheduler import *
import pytorch_lightning as pl
from model.residual_loss import residual_loss_fn
from model.utils import get_loss_fn
from model.msd_mixer import MSDMixer
from torchmetrics import MeanAbsoluteError, MeanSquaredError


class LTFModule(pl.LightningModule):

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
        self.model = MSDMixer(config.seq_len, config.pred_len, config.in_chn,
                              config.ex_chn, config.out_chn, config.patch_sizes,
                              config.hid_len, config.hid_chn, config.hid_pch,
                              config.hid_pred, config.norm, config.use_last_norm,
                              config.activ, config.drop)

        self.val_mse = MeanSquaredError()
        self.val_mae = MeanAbsoluteError()
        self.test_mse = MeanSquaredError()
        self.test_mae = MeanAbsoluteError()
        self.loss_fn = get_loss_fn(config.loss_fn)

    def training_step(self, batch, batch_idx):
        x, y, x_mark, _ = batch
        x = x.float()
        y = y.float()
        x_mark = x_mark.float()
        y = y[:, :, :self.out_chn]
        y_pred, res = self.model(x, x_mark)
        pred_loss = self.loss_fn(y_pred, y)
        residual_loss = residual_loss_fn(res, self.lambda_mse, self.lambda_acf,
                                       self.acf_cutoff)
        loss = pred_loss + residual_loss

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        x, y, x_mark, _ = batch
        x = x.float()
        y = y.float()
        x_mark = x_mark.float()
        y = y[:, :, :self.out_chn]
        y_pred, _ = self.model(x, x_mark)
        self.val_mse(y_pred, y)
        self.val_mae(y_pred, y)
        self.log("val_mse", self.val_mse)
        self.log("val_mae", self.val_mae)
        self.log("lr", self.optimizers().param_groups[0]['lr'])
        return

    def test_step(self, batch, batch_idx):
        x, y, x_mark, _ = batch
        x = x.float()
        y = y.float()
        x_mark = x_mark.float()
        y = y[:, :, :self.out_chn]
        y_pred, _ = self.model(x, x_mark)
        self.test_mse(y_pred, y)
        self.test_mae(y_pred, y)
        self.log("test_mse", self.test_mse)
        self.log("test_mae", self.test_mae)

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
            verbose=False,
            min_lr=1e-8)
        scheduler_config["monitor"] = "val_mse"
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
        }
