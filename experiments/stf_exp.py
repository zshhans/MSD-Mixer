import pytorch_lightning as pl
from configs.stf_config import STFConfig
from task_modules.stf_module import STFModule
from data_provider.data_loader import M4Meta


def stf_experiment(config):
    pl.seed_everything(2021)
    exp = STFModule(config)
    exp.train()
    exp.test()


def run_stf(args):
    dataset_args = set([str.lower(d) for d in args.dataset])
    datasets = []
    if "all" in dataset_args:
        datasets += M4Meta.seasonal_patterns
    else:
        datasets += [d.capitalize() for d in dataset_args]

    for dataset in datasets:
        print(dataset)
        config = STFConfig(dataset)
        stf_experiment(config)
