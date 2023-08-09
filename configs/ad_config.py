class ADConfig():

    def __init__(self) -> None:

        # exp
        self.embed = 'timeF'
        self.task_name = "anomaly_detection"
        self.features = 'M'
        self.target = 'OT'
        self.label_len = 0
        self.seasonal_patterns = "Monthly"
        self.batch_size = 128
        self.num_workers = 8
        self.freq = 'h'
        self.seq_len = 100
        self.pred_len = 0
        self.out_chn = 0
        self.ex_chn = 0

        # model component
        self.activ = "gelu"
        self.norm = None
        self.use_last_norm = False
        self.drop = 0.2

        # loss
        self.loss_fn = 'mse'
        self.lambda_acf = 1
        self.lambda_mse = 0.1
        self.acf_cutoff = 2

        # optim
        self.grad_clip_val = 1
        self.patience = 10
        self.lr = 1e-3
        self.lr_factor = 0.5
        self.optim = 'adam'
        self.weight_decay = 1e-4
        self.max_epochs=50


class SMD_ADConfig(ADConfig):

    def __init__(self) -> None:
        super().__init__()
        self.name = "SMD"
        self.data = "SMD"
        self.root_path = "./dataset/SMD"
        self.in_chn = 38
        self.anomaly_ratio = 0.5

        self.patch_sizes = [50, 25, 10, 5, 1, 1]
        self.hid_chn=512
        self.hid_len=512
        self.hid_pred=512
        self.hid_pch=512
        self.lambda_acf = 1
        self.lambda_mse = 0.01
        self.lr = 3e-4
        self.patience = 3
        self.acf_cutoff = 0


class MSL_ADConfig(ADConfig):

    def __init__(self) -> None:
        super().__init__()
        self.name = "MSL"
        self.data = "MSL"
        self.root_path = "./dataset/MSL"
        self.in_chn = 55
        self.anomaly_ratio = 1

        self.patch_sizes = [20, 10, 5, 2, 1, 1]
        self.hid_chn=512
        self.hid_len=512
        self.hid_pred=512
        self.hid_pch=512


class SMAP_ADConfig(ADConfig):

    def __init__(self) -> None:
        super().__init__()
        self.name = "SMAP"
        self.data = "SMAP"
        self.root_path = "./dataset/SMAP"
        self.in_chn = 25
        self.anomaly_ratio = 1

        self.patch_sizes = [20, 10, 5, 2, 1, 1]
        self.hid_chn=512
        self.hid_len=512
        self.hid_pred=512
        self.hid_pch=512
        self.max_epochs=20


class SWaT_ADConfig(ADConfig):

    def __init__(self) -> None:
        super().__init__()
        self.name = "SWAT"
        self.data = "SWAT"
        self.root_path = "./dataset/SWaT"
        self.in_chn = 51
        self.anomaly_ratio = 1

        self.patch_sizes = [20, 10, 5, 2, 1, 1]
        self.hid_chn=512
        self.hid_len=512
        self.hid_pred=512
        self.hid_pch=512
        self.patience = 1
        self.acf_cutoff = 0
        self.max_epochs=20


class PSM_ADConfig(ADConfig):

    def __init__(self) -> None:
        super().__init__()
        self.name = "PSM"
        self.data = "PSM"
        self.root_path = "./dataset/PSM"
        self.in_chn = 25
        self.anomaly_ratio = 1

        self.patch_sizes = [20, 10, 5, 2, 1, 1]
        self.hid_chn=512
        self.hid_len=512
        self.hid_pred=512
        self.hid_pch=512
        self.patience = 1
        self.acf_cutoff = 0
        self.max_epochs=20
