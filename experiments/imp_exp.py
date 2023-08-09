import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from data_provider.data_factory import data_provider
from task_modules.imp_module import IMPModule
from configs.imp_config import *
from model.utils import get_csv_logger


def imp_experiment(config, gpus):
    pl.seed_everything(2023)
    _, train_dl = data_provider(config, "train")
    _, val_dl = data_provider(config, "val")
    _, test_dl = data_provider(config, "test")

    model = IMPModule(config)
    monitor_metric = "val_mse"
    callbacks = []
    ckpt_callback = ModelCheckpoint(monitor=monitor_metric,
                                    save_top_k=1,
                                    mode="min")
    callbacks.append(ckpt_callback)
    es_callback = EarlyStopping(monitor=monitor_metric,
                                mode="min",
                                patience=10)
    callbacks.append(es_callback)
    logger = get_csv_logger("logs/imp",
                            name=f"{config.name}_{config.mask_rate:.3f}")
    trainer = pl.Trainer(devices=[gpus],
                         accelerator="gpu",
                         precision=16,
                         callbacks=callbacks,
                         logger=logger,
                         max_epochs=40,
                         gradient_clip_val=config.grad_clip_val)
    trainer.fit(model, train_dl, val_dl)
    model = IMPModule.load_from_checkpoint(ckpt_callback.best_model_path)
    trainer.test(model, test_dl)


def run_imp(args):
    dataset_dict = {}
    dataset_dict['ecl'] = ECL_IMPConfig
    dataset_dict['etth1'] = ETTh1_IMPConfig
    dataset_dict['etth2'] = ETTh2_IMPConfig
    dataset_dict['ettm1'] = ETTm1_IMPConfig
    dataset_dict['ettm2'] = ETTm2_IMPConfig
    dataset_dict['weather'] = Weather_IMPConfig
    dataset_args = set([str.lower(d) for d in args.dataset])
    datasets = []
    if "all" in dataset_args:
        datasets += [
            ETTm1_IMPConfig, ETTm2_IMPConfig, ETTh1_IMPConfig, ETTh2_IMPConfig,
            ECL_IMPConfig, Weather_IMPConfig
        ]
    else:
        datasets += [dataset_dict[d] for d in dataset_args]

    mask_rate_args = set([str.lower(p) for p in args.mask_rate])
    mask_rates = []
    if "all" in mask_rate_args:
        mask_rates += [0.125, 0.25, 0.375, 0.5]
    else:
        mask_rates += [float(d) for d in mask_rate_args]

    for dataset in datasets:
        for mask_rate in mask_rates:
            config = dataset(mask_rate)
            print(f"{config.name}_{config.mask_rate:.3f}")
            imp_experiment(config, args.gpus)
