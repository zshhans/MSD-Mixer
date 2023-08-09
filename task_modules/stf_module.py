from configs.stf_config import STFConfig
from data_provider.data_factory import data_provider
from data_provider.m4 import M4Meta
from utils.tools import EarlyStopping
from utils.losses import smape_loss
from utils.m4_summary import M4Summary
from model.residual_loss import residual_loss_fn
from model.msd_mixer import MSDMixer
from torchinfo import summary
from tqdm import tqdm
from pathlib import Path
import os, yaml, torch
import numpy as np
import pandas as pd


class STFModule:

    def __init__(self,
                 config: STFConfig,
                 save_dir="logs",
                 exp_name="stf") -> None:
        self.config = config
        self.config.pred_len = M4Meta.horizons_map[
            self.config.seasonal_patterns]  # Up to M4 config
        self.config.seq_len = 2 * self.config.pred_len  # input_len = 2*pred_len
        self.config.label_len = self.config.pred_len
        self.config.frequency_map = M4Meta.frequency_map[
            self.config.seasonal_patterns]

        if config.use_gpu:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = MSDMixer(config.seq_len, config.pred_len, config.in_chn,
                              config.ex_chn, config.out_chn,
                              config.patch_sizes, config.hid_len,
                              config.hid_chn, config.hid_pch, config.hid_pred,
                              config.norm, config.last_norm, config.activ,
                              config.drop, config.reduction).to(self.device)
        summary(self.model, depth=3)

        self.loss = smape_loss()
        self.exp_dir = Path(save_dir) / exp_name
        self.sub_exp_dir = self.exp_dir / self.config.seasonal_patterns
        os.makedirs(self.sub_exp_dir, exist_ok=True)
        self._save_hparams()

    def _save_hparams(self):
        with open(self.sub_exp_dir / "hparams.yaml", 'w',
                  encoding="utf8") as f:
            yaml.dump(self.config, f, yaml.Dumper)

    def _get_data(self, flag):
        _, data_loader = data_provider(self.config, flag)
        return None, data_loader

    def train(self):
        _, train_loader = self._get_data(flag='train')
        _, vali_loader = self._get_data(flag='val')

        early_stopping = EarlyStopping(patience=self.config.patience,
                                       verbose=True)

        model_optim = torch.optim.Adam(self.model.parameters(),
                                       lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optim,
                                                               'min',
                                                               0.5,
                                                               1,
                                                               verbose=True)

        results = []

        for epoch in range(self.config.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                    tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, res = self.model(batch_x)

                f_dim = -1 if self.config.features == 'MS' else 0
                outputs = outputs[:, -self.config.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.config.pred_len:,
                                  f_dim:].to(self.device)

                batch_y_mark = batch_y_mark[:, -self.config.pred_len:,
                                            f_dim:].to(self.device)
                pred_loss = self.loss(batch_x, self.config.frequency_map,
                                      outputs, batch_y, batch_y_mark)
                res_loss = residual_loss_fn(res, 0., .1)
                loss = pred_loss + res_loss
                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()

            train_loss = np.average(train_loss)
            vali_loss = self.vali(train_loader, vali_loader)
            results.append({
                'Epoch': epoch,
                'train_loss': train_loss,
                "val_loss": vali_loss,
                "lr": model_optim.param_groups[0]['lr']
            })
            print(f"epoch: {epoch}, train_loss: {train_loss:.3f}, " +
                  f"val_loss: {vali_loss:.3f}, " +
                  f"lr: {model_optim.param_groups[0]['lr']}")
            early_stopping(vali_loss, self.model, str(self.sub_exp_dir))
            if early_stopping.early_stop:
                break
            scheduler.step(vali_loss)

        df = pd.DataFrame(results)
        df.to_csv(self.sub_exp_dir / 'metrics.csv', index=False)
        best_model_path = self.sub_exp_dir / 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

    def vali(self, train_loader, vali_loader):
        x, _ = train_loader.dataset.last_insample_window()
        y = vali_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)

        self.model.eval()
        with torch.no_grad():
            B, _, C = x.shape
            outputs = torch.zeros((B, self.config.pred_len, C)).float()
            id_list = np.arange(0, B, 500)
            id_list = np.append(id_list, B)
            for i in range(len(id_list) - 1):
                outputs[id_list[i]:id_list[i + 1], :, :] = self.model(
                    x[id_list[i]:id_list[i + 1]])[0].detach().cpu()
            f_dim = -1 if self.config.features == 'MS' else 0
            outputs = outputs[:, -self.config.pred_len:, f_dim:]
            pred = outputs
            true = torch.from_numpy(np.array(y))
            batch_y_mark = torch.ones(true.shape)

            loss = self.loss(x.detach().cpu()[:, :,
                                              0], self.config.frequency_map,
                             pred[:, :, 0], true, batch_y_mark)

        self.model.train()
        return loss

    def test(self):
        _, train_loader = self._get_data(flag='train')
        _, test_loader = self._get_data(flag='test')
        x, _ = train_loader.dataset.last_insample_window()
        y = test_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)

        self.model.eval()
        with torch.no_grad():
            B, _, C = x.shape
            outputs = torch.zeros(
                (B, self.config.pred_len, C)).float().to(self.device)
            id_list = np.arange(0, B, 1)
            id_list = np.append(id_list, B)
            for i in range(len(id_list) - 1):
                outputs[id_list[i]:id_list[i + 1], :, :] = self.model(
                    x[id_list[i]:id_list[i + 1]])[0]

            f_dim = -1 if self.config.features == 'MS' else 0
            outputs = outputs[:, -self.config.pred_len:, f_dim:]
            outputs = outputs.detach().cpu().numpy()

            preds = outputs
            trues = y
            x = x.detach().cpu().numpy()

        forecasts_df = pd.DataFrame(
            preds[:, :, 0],
            columns=[f'V{i + 1}' for i in range(self.config.pred_len)])
        forecasts_df.index = test_loader.dataset.ids[:preds.shape[0]]
        forecasts_df.index.name = 'id'
        forecasts_df.set_index(forecasts_df.columns[0], inplace=True)
        forecast_filename = f'{self.config.seasonal_patterns}_forecast.csv'
        forecasts_df.to_csv(self.exp_dir / forecast_filename)

        for sp in M4Meta.seasonal_patterns:
            if not os.path.exists(self.exp_dir / f'{sp}_forecast.csv'):
                return

        m4_summary = M4Summary(str(self.exp_dir) + "/", self.config.root_path)
        smape, owa, _, mase = m4_summary.evaluate()
        smape['metric'] = "smape"
        mase['metric'] = "mase"
        owa['metric'] = "owa"
        df = pd.DataFrame([smape, mase, owa])
        print(df.set_index('metric'))
        df.to_csv(self.exp_dir / 'avg_metrics.csv', index=False)
