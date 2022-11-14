""" usage: python run_test_dataset.py --config-file /home/lyltc/git/GDR-Net/configs/symn/ycbv/symn_net_config.py
    In this script, we set TEST datasets to ("test",) , detail in # debug setting.
"""
import math
import numpy as np
from mmcv import Config
from core.utils.default_args_setup import my_default_argument_parser
from lib.utils.utils import iprint
from core.symn.datasets.BOPDataset_utils import build_BOP_det_crop_dataset
import cv2

parser = my_default_argument_parser()
args = parser.parse_args()
cfg = Config.fromfile(args.config_file)

# debug setting
iprint("DEBUG")
args.num_gpus = 1
args.num_machines = 1
cfg.TRAIN.NUM_WORKERS = 0
cfg.TRAIN.PRINT_FREQ = 1
cfg.TRAIN.IMS_PER_BATCH = 2
cfg.DATASETS.TEST = "test"


# ---- main function build datasets, data is test datasets
data = build_BOP_det_crop_dataset(cfg)

obj_ids = cfg.DATASETS.OBJ_IDS
res_crop = cfg.DATASETS.RES_CROP
window_names = ['rgb_crop']
for j, name in enumerate(window_names):
    cv2.namedWindow(name, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(name, res_crop, res_crop)
    cv2.moveWindow(name, 1 + 250 * j, 1 + 250 * 0)

def denormalize(img):
    mu, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    img = (img.transpose([1, 2, 0]) * std + mu) * 255
    return img.astype(np.uint8)

print()
print('With an opencv window active:')
print("press 'a', 'd' and 'x'(random) to get a new input image,")
print("press 'q' to quit.")
data_i = 0
while True:
    print()
    print('------------ new input -------------')
    inst = data[data_i]
    obj_idx = inst['obj_idx']
    print(f'i: {data_i}, obj_id: {obj_ids[obj_idx]}')
    rgb_crop = inst['rgb_crop']
    rgb_crop = denormalize(rgb_crop)
    cv2.imshow('rgb_crop', rgb_crop[..., ::-1])

    while True:
        print()
        key = cv2.waitKey()
        if key == ord('q'):
            cv2.destroyAllWindows()
            quit()
        elif key == ord('a'):
            data_i = (data_i - 1) % len(data)
            break
        elif key == ord('d'):
            data_i = (data_i + 1) % len(data)
            break
        elif key == ord('x'):
            data_i = np.random.randint(len(data))
            break
