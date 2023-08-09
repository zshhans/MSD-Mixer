from data_provider.data_factory import data_provider


class CLSConfig():

    def __init__(self, data_name) -> None:
        self.name = data_name
        self.embed = 'timeF'
        self.task_name = "classification"
        self.data = 'UEA'
        self.root_path = f"./dataset/Multivariate_ts/{data_name}/"
        self.batch_size = 128
        self.features = 'M'
        self.target = 'OT'
        self.freq = 'h'
        self.seasonal_patterns = "Monthly"
        self.num_workers = 8
        self.norm = 'in'
        self.data_name = data_name

        if data_name == "ArticularyWordRecognition":
            self.use_last_norm = False
            self.drop = 0.2
            self.patch_sizes = [8, 4, 2, 1, 1]
            self.hid_chn = 512
            self.hid_len = 512
            self.hid_pred = 512
            self.hid_pch = 512
            self.lambda_acf = 0.5
            self.lambda_mse = 0.5
            self.acf_cutoff = 0
            self.optim = "adam"
            self.weight_decay = 1e-4
            self.patience = 10
            self.lr = 3e-4
            self.lr_factor = 0.5
            self.activ = "gelu"
            self.reduction = "sum"
            self.grad_clip_val = 0.5
            self.norm = "in"
            self.batch_size = 4

        if data_name == "AtrialFibrillation":
            self.use_last_norm = False

            self.drop = 0.2
            self.patch_sizes = [1, 1, 1]
            self.hid_chn = 512
            self.hid_len = 512
            self.hid_pred = 512
            self.hid_pch = 512
            self.lambda_acf = 0.5
            self.lambda_mse = 0.5
            self.acf_cutoff = 0
            self.optim = "adam"
            self.weight_decay = 1e-4
            self.patience = 10
            self.lr = 1e-3
            self.lr_factor = 0.5
            self.activ = "gelu"
            self.reduction = "max"
            self.grad_clip_val = 0.5
            self.norm = None
            self.batch_size = 4

        if data_name == "CharacterTrajectories":
            self.use_last_norm = False

            self.drop = 0.2
            self.patch_sizes = [1, 1, 1]
            self.hid_chn = 512
            self.hid_len = 512
            self.hid_pred = 512
            self.hid_pch = 512
            self.lambda_acf = 0.5
            self.lambda_mse = 0
            self.acf_cutoff = 2
            self.optim = "adam"
            self.weight_decay = 1e-4
            self.patience = 10
            self.lr = 1e-3
            self.lr_factor = 0.5
            self.activ = "gelu"
            self.reduction = "max"
            self.grad_clip_val = 1
            self.norm = None
            self.batch_size = 4

        if data_name == "Cricket":
            self.use_last_norm = True

            self.drop = 0.5
            self.patch_sizes = [1, 1, 1, 1, 1]
            self.hid_chn = 128
            self.hid_len = 128
            self.hid_pred = 128
            self.hid_pch = 128
            self.lambda_acf = 0.5
            self.lambda_mse = 0
            self.acf_cutoff = 2
            self.optim = "adamw"
            self.weight_decay = 1e-4
            self.patience = 10
            self.lr = 1e-3
            self.lr_factor = 0.5
            self.activ = "gelu"
            self.reduction = "sum"
            self.grad_clip_val = 0.5
            self.norm = "bn"
            self.batch_size = 9

        if data_name == "FaceDetection":
            self.use_last_norm = False

            self.drop = 0.5
            self.patch_sizes = [1, 1, 1, 1, 1]
            self.hid_chn = 512
            self.hid_len = 512
            self.hid_pred = 512
            self.hid_pch = 512
            self.lambda_acf = 0.5
            self.lambda_mse = 0.5
            self.acf_cutoff = 1
            self.optim = "adam"
            self.weight_decay = 1e-4
            self.patience = 10
            self.lr = 3e-4
            self.lr_factor = 0.5
            self.activ = "gelu"
            self.reduction = "sum"
            self.grad_clip_val = 1
            self.norm = None
            self.batch_size = 4

        if data_name == "FingerMovements":
            self.use_last_norm = True

            self.drop = 0.5
            self.patch_sizes = [1, 1, 1]
            self.hid_chn = 128
            self.hid_len = 128
            self.hid_pred = 128
            self.hid_pch = 128
            self.lambda_acf = 0.5
            self.lambda_mse = 0.5
            self.acf_cutoff = 1
            self.optim = "adam"
            self.weight_decay = 1
            self.patience = 10
            self.lr = 3e-4
            self.lr_factor = 0.5
            self.activ = "gelu"
            self.reduction = "max"
            self.grad_clip_val = 0.5
            self.norm = None
            self.batch_size = 4
        
        if data_name == "MotorImagery":
            self.use_last_norm = True

            self.drop = 0.5
            self.patch_sizes = [1, 1, 1]
            self.hid_chn = 256
            self.hid_len = 256
            self.hid_pred = 256
            self.hid_pch = 256
            self.lambda_acf = 0.5
            self.lambda_mse = 0.5
            self.acf_cutoff = 0
            self.optim = "adam"
            self.weight_decay = 1e-4
            self.patience = 10
            self.lr = 1e-3
            self.lr_factor = 0.5
            self.activ = "gelu"
            self.reduction = "max"
            self.grad_clip_val = 1
            self.norm = None
            self.batch_size = 1

        if data_name == "SelfRegulationSCP1":
            self.use_last_norm = False

            self.drop = 0.2
            self.patch_sizes = [1, 1, 1, 1, 1]
            self.hid_chn = 512
            self.hid_len = 512
            self.hid_pred = 512
            self.hid_pch = 512
            self.lambda_acf = 0.5
            self.lambda_mse = 0
            self.acf_cutoff = 0
            self.optim = "adam"
            self.weight_decay = 1e-4
            self.patience = 10
            self.lr = 3e-4
            self.lr_factor = 0.5
            self.activ = "gelu"
            self.reduction = "sum"
            self.grad_clip_val = 1
            self.norm = None
            self.batch_size = 4

        if data_name == "SelfRegulationSCP2":
            self.use_last_norm = True

            self.drop = 0.5
            self.patch_sizes = [1, 1, 1, 1, 1]
            self.hid_chn = 128
            self.hid_len = 128
            self.hid_pred = 128
            self.hid_pch = 128
            self.lambda_acf = 0.5
            self.lambda_mse = 0
            self.acf_cutoff = 2
            self.optim = "adam"
            self.weight_decay = 1e-4
            self.patience = 10
            self.lr = 3e-4
            self.lr_factor = 0.5
            self.activ = "gelu"
            self.reduction = "max"
            self.grad_clip_val = 1
            self.norm = None
            self.batch_size = 4

        if data_name == "UWaveGestureLibrary":
            self.use_last_norm = True

            self.drop = 0.2
            self.patch_sizes = [8, 4, 2, 1, 1]
            self.hid_chn = 512
            self.hid_len = 512
            self.hid_pred = 512
            self.hid_pch = 512
            self.lambda_acf = 0.5
            self.lambda_mse = 0.5
            self.acf_cutoff = 1
            self.optim = "adam"
            self.weight_decay = 1e-4
            self.patience = 10
            self.lr = 3e-4
            self.lr_factor = 0.5
            self.activ = "gelu"
            self.reduction = "sum"
            self.grad_clip_val = 0.5
            self.norm = None
            self.batch_size = 4

        self.train_ds, self.train_dl = data_provider(self, "TRAIN")
        self.test_ds, self.test_dl = data_provider(self, "TEST")
        self.seq_len = int(
            max(self.train_ds.max_seq_len, self.test_ds.max_seq_len))
        self.pred_len = 1
        self.in_chn = self.train_ds.feature_df.shape[1]
        self.out_chn = len(self.train_ds.class_names)
        self.ex_chn = 0
