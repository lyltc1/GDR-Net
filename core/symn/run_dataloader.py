""" usage: python run_dataloader.py --config-file /home/lyltc/git/GDR-Net/configs/symn/ycbv/symn_net_config.py

"""
from torch import Tensor
from mmcv import Config
from core.utils.default_args_setup import my_default_argument_parser
from lib.utils.utils import iprint
from core.symn.datasets.BOPDataset_utils import *

parser = my_default_argument_parser()
args = parser.parse_args()
cfg = Config.fromfile(args.config_file)

# debug setting
iprint("DEBUG: build ")
args.num_gpus = 1
args.num_machines = 1
cfg.TRAIN.NUM_WORKERS = 0
cfg.TRAIN.PRINT_FREQ = 1
cfg.TRAIN.IMS_PER_BATCH = 4
cfg.DATASETS.TRAIN = ("real",)
cfg.DATASETS.TEST = "test"

# ---- build train_data_loader
train_dataset = build_train_BOP_instance_dataset(cfg, cfg.DATASETS.TRAIN)
train_data_loader = build_train_dataloader(cfg, train_dataset)
train_data_loader_iter = iter(train_data_loader)
train_dataset_len = len(train_data_loader.dataset)
train_inst = next(train_data_loader_iter)
for key, value in train_inst.items():
    if isinstance(value, Tensor):
        print(f"{key}:{type(value)}-{value.shape}")
    else:
        print(f"{key}:{type(value)}")

# ---- build train_data_loader
data_test = build_BOP_det_crop_dataset(cfg)
test_data_loader = build_test_dataloader(cfg, data_test)
test_data_loader_iter = iter(test_data_loader)
test_dataset_len = len(test_data_loader.dataset)
test_inst = next(test_data_loader_iter)
for key, value in train_inst.items():
    if isinstance(value, Tensor):
        print(f"{key}:{type(value)}-{value.shape}")
    else:
        print(f"{key}:{type(value)}")
