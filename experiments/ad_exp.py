import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from data_provider.data_factory import data_provider
from task_modules.ad_module import ADModule
from configs.ad_config import *
from model.utils import get_csv_logger


def ad_experiment(config, gpus):
    pl.seed_everything(2023)
    _, train_dl = data_provider(config, "train")
    _, val_dl = data_provider(config, "val")
    _, test_dl = data_provider(config, "test")
    monitor_metric = "val_mse"
    monitor_mode = "min"
    test_dls = [train_dl, test_dl]

    model = ADModule(config)
    callbacks = []
    ckpt_callback = ModelCheckpoint(monitor=monitor_metric,
                                    save_top_k=1,
                                    mode=monitor_mode)
    callbacks.append(ckpt_callback)
    es_callback = EarlyStopping(monitor=monitor_metric,
                                mode=monitor_mode,
                                patience=12)
    callbacks.append(es_callback)
    logger = get_csv_logger("logs/ad", name=f"{config.name}")
    trainer = pl.Trainer(devices=[gpus],
                         accelerator="gpu",
                         precision=16,
                         callbacks=callbacks,
                         logger=logger,
                         max_epochs=config.max_epochs,
                         gradient_clip_val=config.grad_clip_val)
    trainer.fit(model, train_dl, val_dl)
    model = ADModule.load_from_checkpoint(ckpt_callback.best_model_path)
    trainer.test(model, test_dls)


def run_ad(args):
    dataset_dict = {}
    dataset_dict['smd'] = SMD_ADConfig
    dataset_dict['msl'] = MSL_ADConfig
    dataset_dict['smap'] = SMAP_ADConfig
    dataset_dict['swat'] = SWaT_ADConfig
    dataset_dict['psm'] = PSM_ADConfig
    dataset_args = set([str.lower(d) for d in args.dataset])
    datasets = []
    if "all" in dataset_args:
        datasets += [
            SMD_ADConfig, MSL_ADConfig, SMAP_ADConfig, SWaT_ADConfig,
            PSM_ADConfig
        ]
    else:
        datasets += [dataset_dict[d] for d in dataset_args]

    for dataset in datasets:
        config = dataset()
        print(config.name)
        ad_experiment(config, args.gpus)
