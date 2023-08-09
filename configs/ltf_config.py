class LTFConfig():

    def __init__(self, pred_len) -> None:

        # exp
        self.embed = 'timeF'
        self.task_name = "long_term_forecast"
        self.features = 'M'
        self.target = 'OT'
        self.label_len = 0
        self.seasonal_patterns = "Monthly"
        self.batch_size = 32
        self.num_workers = 8
        self.freq = 'h'
        self.ex_chn = 4
        self.seq_len = 96
        self.pred_len = pred_len

        # model component
        self.activ = "gelu"
        self.norm = None
        self.use_last_norm = True

        # loss
        self.loss_fn = 'mse'
        self.lambda_acf = 0.5
        self.lambda_mse = 0.1
        self.acf_cutoff = 2

        # optim
        self.grad_clip_val = 1
        self.patience = 1
        self.lr = 1e-3
        self.lr_factor = 0.5
        self.optim = "adamw"
        self.weight_decay = 0.1


class ECL_LTFConfig(LTFConfig):

    def __init__(self, pred_len) -> None:
        super().__init__(pred_len)
        self.name = "ecl"
        self.data = "custom"
        self.root_path = "./dataset/electricity/"
        self.data_path = "electricity.csv"
        self.in_chn = 321
        self.out_chn = 321

        self.drop = 0.5
        self.patch_sizes = [24, 12, 6, 2, 1]
        self.hid_chn = 256
        self.hid_len = 256
        self.hid_pred = 256
        self.hid_pch = 256


class ETTm1_LTFConfig(LTFConfig):

    def __init__(self, pred_len) -> None:
        super().__init__(pred_len)
        self.name = "ettm1"
        self.data = "ETTm1"
        self.root_path = "./dataset/ETT-small/"
        self.data_path = "ETTm1.csv"
        self.in_chn = 7
        self.out_chn = 7

        self.drop = 0.5
        self.patch_sizes = [24, 12, 4, 2, 1]
        self.hid_chn = 512
        self.hid_len = 512
        self.hid_pred = 512
        self.hid_pch = 512


class ETTm2_LTFConfig(LTFConfig):

    def __init__(self, pred_len) -> None:
        super().__init__(pred_len)
        self.name = "ettm2"
        self.data = "ETTm2"
        self.root_path = "./dataset/ETT-small/"
        self.data_path = "ETTm2.csv"
        self.in_chn = 7
        self.out_chn = 7

        self.drop = 0.5
        self.patch_sizes = [24, 12, 4, 2, 1]
        self.hid_chn = 512
        self.hid_len = 512
        self.hid_pred = 512
        self.hid_pch = 512


class ETTh1_LTFConfig(LTFConfig):

    def __init__(self, pred_len) -> None:
        super().__init__(pred_len)
        self.name = "etth1"
        self.data = "ETTh1"
        self.root_path = "./dataset/ETT-small/"
        self.data_path = "ETTh1.csv"
        self.in_chn = 7
        self.out_chn = 7

        self.drop = 0.5
        self.patch_sizes = [24, 12, 6, 2, 1]
        self.hid_chn = 512
        self.hid_len = 512
        self.hid_pred = 512
        self.hid_pch = 512
        self.weight_decay = 1


class ETTh2_LTFConfig(LTFConfig):

    def __init__(self, pred_len) -> None:
        super().__init__(pred_len)
        self.name = "etth2"
        self.data = "ETTh2"
        self.root_path = "./dataset/ETT-small/"
        self.data_path = "ETTh2.csv"
        self.in_chn = 7
        self.out_chn = 7

        self.drop = 0.5
        self.patch_sizes = [24, 12, 6, 2, 1]
        self.hid_chn = 512
        self.hid_len = 512
        self.hid_pred = 512
        self.hid_pch = 512


class Traffic_LTFConfig(LTFConfig):

    def __init__(self, pred_len) -> None:
        super().__init__(pred_len)
        self.name = "traffic"
        self.data = "custom"
        self.root_path = "./dataset/traffic/"
        self.data_path = "traffic.csv"
        self.in_chn = 862
        self.out_chn = 862
        self.batch_size = 16

        self.drop = 0.5
        self.patch_sizes = [24, 12, 6, 1]
        self.hid_chn = 512
        self.hid_len = 512
        self.hid_pred = 512
        self.hid_pch = 512


class Weather_LTFConfig(LTFConfig):

    def __init__(self, pred_len) -> None:
        super().__init__(pred_len)
        self.name = "weather"
        self.data = "custom"
        self.root_path = "./dataset/weather/"
        self.data_path = "weather.csv"
        self.in_chn = 21
        self.out_chn = 21

        self.drop = 0.4
        self.patch_sizes = [12, 6, 3, 1, 1]
        self.hid_chn = 256
        self.hid_len = 256
        self.hid_pred = 256
        self.hid_pch = 256


class Exchange_LTFConfig(LTFConfig):

    def __init__(self, pred_len) -> None:
        super().__init__(pred_len)
        self.name = "exchange"
        self.data = "custom"
        self.root_path = "./dataset/exchange_rate/"
        self.data_path = "exchange_rate.csv"
        self.in_chn = 8
        self.out_chn = 8

        self.drop = 0.5
        self.patch_sizes = [28, 14, 7, 2, 1]
        self.hid_chn = 256
        self.hid_len = 256
        self.hid_pred = 256
        self.hid_pch = 256
        self.lr = 3e-5
