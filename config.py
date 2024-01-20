import os
import torch

class ArgsConfig:
    def __init__(self) -> None:
        self.batch_size = 192
        self.embedding_size = 480
        self.epochs = 100
        self.kflod = 5
        self.max_len = 51
        self.lr = 1.8e-3
        self.weight_decay = 0
        self.dropout = 0.6
        self.ctl = False
        
        self.margin = 2.8
        self.scale_factor = 1
        self.training_ratio = 1.1
        self.model_name = 'DeepMFPP-MFTP'
        self.loss_fn_name = 'MLFDL'
        self.exp_nums = None
        self.aa_dict = None # 'protbert' /'esm'/ None
        self.class_weight = False
        
        self.fldl_clip_pos = 0.7
        self.fldl_clip_neg = 0.5
        self.fldl_pos_weight = 0.4
        self.info = f"FDL{0.7,0.5,0.3},CosLR,cw={self.class_weight},使用自己的evaluation"  #对当前训练做的补充说明

        self.data_dir = './data/AllData.txt'
        self.train_data_dir = './data/MFTP-Data/train.txt' # MLBP-Data/train.txt  MFTP-Data/train.txt
        self.test_data_dir = './data/MFTP-Data/test.txt' # MLBP-Data/test.txt  MFTP-Data/test.txt
        # MLBP-Data/train_0.5_min-2_maj-1.txt 
        # MFTP-Data/traindata_da/train_rs_2.txt
        self.train_data_da_dir = './data/MFTP-Data/traindata_da/train_rs_2.txt' 
        self.ebv_dir = './eq_21_21.pkl'
        self.use_ebv = False

        self.log_dir = './result/logs'
        self.save_dir = './result/model_para'
        self.tensorboard_log_dir = './tensorboard'
        self.ems_path = './ESM2/esm2_t12_35M_UR50D.pt'
        self.esm_layer_idx = 12
        self.save_para_dir = os.path.join(self.save_dir,self.model_name)
        self.random_seed = 2022
        self.num_classes = 21
        self.split_size = 0.8
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.continue_training = False
        self.checkpoint_path = " "
        
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        if not os.path.exists(self.save_para_dir):
            os.mkdir(self.save_para_dir)
        if not os.path.exists(self.tensorboard_log_dir):
            os.mkdir(self.tensorboard_log_dir)