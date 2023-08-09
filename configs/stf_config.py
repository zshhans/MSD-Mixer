class STFConfig():
    def __init__(self,seasonal_patterns) -> None:
        self.task_name="short_term_forecast"
        self.use_gpu=True
        self.data="m4"
        self.root_path="./dataset/m4"
        self.data_path=""
        self.num_workers=10
        self.seasonal_patterns=seasonal_patterns
        self.target="OT"
        self.features="M"
        self.freq="h"
        self.embed = 'timeF'
        self.in_chn=1
        self.ex_chn=0
        self.out_chn=1
        self.itr=1
        self.batch_size=16
        self.patience=3
        self.learning_rate=3e-4
        self.train_epochs=30
        
        self.reduction = 'sum'
        self.norm = None
        self.last_norm = True
        self.drop = 0
        self.activ = "gelu"
        

        if seasonal_patterns == "Yearly":
            self.patch_sizes = [3, 1, 1, 1]
            self.hid_len = 512
            self.hid_chn = 128
            self.hid_pch = 128
            self.hid_pred = 512
            self.learning_rate=1e-3

        elif seasonal_patterns == "Quarterly":
            self.patch_sizes = [4, 4, 1, 1]
            self.hid_len = 512
            self.hid_chn = 128
            self.hid_pch = 128
            self.hid_pred = 512

        elif seasonal_patterns == "Monthly":
            self.patch_sizes = [12, 6, 3, 1, 1]
            self.hid_len = 512
            self.hid_chn = 128
            self.hid_pch = 128
            self.hid_pred = 512

        elif seasonal_patterns == "Weekly":
            self.patch_sizes = [12, 4, 4, 1, 1]
            self.hid_len = 64
            self.hid_chn = 32
            self.hid_pch = 32
            self.hid_pred = 64

        elif seasonal_patterns == "Daily":
            self.patch_sizes = [14, 7, 7, 1, 1]
            self.hid_len = 64
            self.hid_chn = 32
            self.hid_pch = 32
            self.hid_pred = 64

        elif seasonal_patterns == "Hourly":
            self.patch_sizes = [12, 6, 6, 1, 1]
            self.hid_len = 64
            self.hid_chn = 32
            self.hid_pch = 32
            self.hid_pred = 64