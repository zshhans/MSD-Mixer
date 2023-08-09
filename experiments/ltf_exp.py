import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from data_provider.data_factory import data_provider
from task_modules.ltf_module import LTFModule
from configs.ltf_config import *
from model.utils import get_csv_logger


def ltf_experiment(config, gpus):
    pl.seed_everything(2023)
    _, train_dl = data_provider(config, "train")
    _, val_dl = data_provider(config, "val")
    _, test_dl = data_provider(config, "test")

    model = LTFModule(config)
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
    logger = get_csv_logger("logs/ltf",
                            name=f"{config.name}_{config.pred_len}")
    trainer = pl.Trainer(devices=[gpus],
                         accelerator="gpu",
                         precision=16,
                         callbacks=callbacks,
                         logger=logger,
                         max_epochs=50,
                         gradient_clip_val=config.grad_clip_val)
    trainer.fit(model, train_dl, val_dl)
    model = LTFModule.load_from_checkpoint(ckpt_callback.best_model_path)
    trainer.test(model, test_dl)


def run_ltf(args):
    dataset_dict = {}
    dataset_dict['ecl'] = ECL_LTFConfig
    dataset_dict['etth1'] = ETTh1_LTFConfig
    dataset_dict['etth2'] = ETTh2_LTFConfig
    dataset_dict['ettm1'] = ETTm1_LTFConfig
    dataset_dict['ettm2'] = ETTm2_LTFConfig
    dataset_dict['traffic'] = Traffic_LTFConfig
    dataset_dict['weather'] = Weather_LTFConfig
    dataset_dict['exchange'] = Exchange_LTFConfig
    dataset_args = set([str.lower(d) for d in args.dataset])
    datasets = []
    if "all" in dataset_args:
        datasets += [
            ECL_LTFConfig, ETTh1_LTFConfig, ETTh2_LTFConfig, ETTm1_LTFConfig,
            ETTm2_LTFConfig, Traffic_LTFConfig, Weather_LTFConfig,
            Exchange_LTFConfig
        ]
    else:
        datasets += [dataset_dict[d] for d in dataset_args]

    pred_len_args = set([str.lower(p) for p in args.pred_len])
    pred_lens = []
    if "all" in pred_len_args:
        pred_lens += [96, 192, 336, 720]
    else:
        pred_lens += [int(d) for d in pred_len_args]

    for dataset in datasets:
        for pred_len in pred_lens:
            config = dataset(pred_len)
            print(f"{config.name}_{config.pred_len}")
            ltf_experiment(config, args.gpus)
