class IMPConfig():

    def __init__(self, mask_rate) -> None:

        # exp
        self.embed = 'timeF'
        self.task_name = "long_term_forecast"
        self.features = 'M'
        self.target = 'OT'
        self.label_len = 0
        self.seasonal_patterns = "Monthly"
        self.batch_size = 16
        self.num_workers = 8
        self.freq = 'h'
        self.ex_chn = 4
        self.pred_len = 96
        self.mask_rate = mask_rate

        # model component
        self.activ = "gelu"
        self.norm = None
        self.use_last_norm = False
        self.drop = 0.2

        # loss
        self.loss_fn = 'mse'
        self.lambda_acf = 0
        self.lambda_mse = 1
        self.acf_cutoff = 2

        # optim
        self.grad_clip_val = 1
        self.patience = 1
        self.lr = 1e-3
        self.lr_factor = 0.5
        self.optim = 'adam'
        self.weight_decay = 0


class ECL_IMPConfig(IMPConfig):

    def __init__(self, mask_rate) -> None:
        super().__init__(mask_rate)
        self.name = "ecl"
        self.data = "custom"
        self.root_path = "./dataset/electricity/"
        self.data_path = "electricity.csv"
        self.seq_len = 96
        self.in_chn = 321
        self.out_chn = 321

        self.patch_sizes = [24, 12, 6, 2, 1]
        self.hid_chn = 512
        self.hid_len = 512
        self.hid_pred = 512
        self.hid_pch = 512


class ETTm1_IMPConfig(IMPConfig):

    def __init__(self, mask_rate) -> None:
        super().__init__(mask_rate)
        self.name = "ettm1"
        self.data = "ETTm1"
        self.root_path = "./dataset/ETT-small/"
        self.data_path = "ETTm1.csv"
        self.seq_len = 96
        self.in_chn = 7
        self.out_chn = 7

        self.patch_sizes = [8, 4, 2, 1, 1, 1]
        self.hid_chn = 512
        self.hid_len = 512
        self.hid_pred = 512
        self.hid_pch = 512


class ETTm2_IMPConfig(IMPConfig):

    def __init__(self, mask_rate) -> None:
        super().__init__(mask_rate)
        self.name = "ettm2"
        self.data = "ETTm2"
        self.root_path = "./dataset/ETT-small/"
        self.data_path = "ETTm2.csv"
        self.seq_len = 96
        self.in_chn = 7
        self.out_chn = 7

        self.patch_sizes = [24, 16, 8, 4, 1, 1]
        self.hid_chn = 512
        self.hid_len = 512
        self.hid_pred = 512
        self.hid_pch = 512
        self.patience = 0
        self.weight_decay = 1e-4


class ETTh1_IMPConfig(IMPConfig):

    def __init__(self, mask_rate) -> None:
        super().__init__(mask_rate)
        self.name = "etth1"
        self.data = "ETTh1"
        self.root_path = "./dataset/ETT-small/"
        self.data_path = "ETTh1.csv"
        self.seq_len = 96
        self.in_chn = 7
        self.out_chn = 7

        self.patch_sizes = [12, 6, 1, 1, 1]
        self.hid_chn = 512
        self.hid_len = 512
        self.hid_pred = 512
        self.hid_pch = 512


class ETTh2_IMPConfig(IMPConfig):

    def __init__(self, mask_rate) -> None:
        super().__init__(mask_rate)
        self.name = "etth2"
        self.data = "ETTh2"
        self.root_path = "./dataset/ETT-small/"
        self.data_path = "ETTh2.csv"
        self.seq_len = 96
        self.in_chn = 7
        self.out_chn = 7

        self.patch_sizes = [12, 6, 1, 1, 1]
        self.hid_chn = 512
        self.hid_len = 512
        self.hid_pred = 512
        self.hid_pch = 512
        self.weight_decay = 1e-4


class Weather_IMPConfig(IMPConfig):

    def __init__(self, mask_rate) -> None:
        super().__init__(mask_rate)
        self.name = "weather"
        self.data = "custom"
        self.root_path = "./dataset/weather/"
        self.data_path = "weather.csv"
        self.seq_len = 96
        self.in_chn = 21
        self.out_chn = 21

        self.patch_sizes = [24, 12, 6, 1, 1]
        self.hid_chn = 512
        self.hid_len = 512
        self.hid_pred = 512
        self.hid_pch = 512
        self.weight_decay = 1e-4
