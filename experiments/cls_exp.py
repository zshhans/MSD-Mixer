import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from task_modules.cls_module import CLSModule
from configs.cls_config import *
from model.utils import get_csv_logger


def cls_experiment(config, gpus):
    pl.seed_everything(2023)
    model = CLSModule(config.seq_len, config.pred_len, config.in_chn,
                      config.out_chn, config.ex_chn, config.patch_sizes,
                      config.hid_len, config.hid_chn, config.hid_pch,
                      config.hid_pred, config.norm, config.use_last_norm,
                      config.drop, config.lambda_mse, config.lambda_acf,
                      config.acf_cutoff, config.optim, config.lr,
                      config.lr_factor, config.weight_decay, config.patience,
                      config.activ, config.reduction, config.batch_size)
    callbacks = []
    ckpt_callback = ModelCheckpoint(monitor="val_acc",
                                    save_top_k=1,
                                    mode="max")
    callbacks.append(ckpt_callback)
    es_callback = EarlyStopping(monitor="val_acc", mode="max", patience=30)
    callbacks.append(es_callback)
    logger = get_csv_logger("logs/cls", name=f"{config.name}")
    trainer = pl.Trainer(devices=[gpus],
                         accelerator="gpu",
                         precision=16,
                         callbacks=callbacks,
                         logger=logger,
                         max_epochs=100,
                         gradient_clip_val=config.grad_clip_val)
    trainer.fit(model, config.train_dl, config.test_dl)
    model = CLSModule.load_from_checkpoint(ckpt_callback.best_model_path)
    trainer.test(model, config.test_dl)


def run_cls(args):
    dataset_dict = {}
    dataset_dict['awr'] = "ArticularyWordRecognition"
    dataset_dict['af'] = 'AtrialFibrillation'
    dataset_dict['ct'] = 'CharacterTrajectories'
    dataset_dict['cr'] = 'Cricket'
    dataset_dict['fd'] = 'FaceDetection'
    dataset_dict['fm'] = 'FingerMovements'
    dataset_dict['mi'] = 'MotorImagery'
    dataset_dict['scp1'] = 'SelfRegulationSCP1'
    dataset_dict['scp2'] = 'SelfRegulationSCP2'
    dataset_dict['uwgl'] = 'UWaveGestureLibrary'
    dataset_args = set([str.lower(d) for d in args.dataset])
    datasets = []
    if "all" in dataset_args:
        datasets += [
            "ArticularyWordRecognition", 'AtrialFibrillation',
            'CharacterTrajectories', 'Cricket', 'FaceDetection',
            'FingerMovements', 'MotorImagery', 'SelfRegulationSCP1',
            'SelfRegulationSCP2', 'UWaveGestureLibrary'
        ]
    else:
        datasets += [dataset_dict[d] for d in dataset_args]
    for dataset in datasets:
        config=CLSConfig(dataset)
        print(config.name)
        cls_experiment(config,args.gpus)
