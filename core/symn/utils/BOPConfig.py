""" BOP config """

import os
from pathlib import Path
from collections import defaultdict
root = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

class DatasetConfig:
    model_folder = 'models'
    train_folder = 'train_real'
    test_folder = 'test'
    img_folder = 'rgb'
    depth_folder = 'depth'
    img_ext = 'png'
    depth_ext = 'png'
    models_GT_color_folder = root / 'datasets' / 'models_GT_color_v3'
    binary_code_folder = root / 'datasets' / 'binary_code_v3'


bopConfig = defaultdict(lambda *_: DatasetConfig())

bopConfig['tless'] = tless = DatasetConfig()
tless.model_folder = 'models_cad'
tless.test_folder = 'test_primesense'
tless.train_folder = 'train_primesense'

bopConfig['hb'] = hb = DatasetConfig()
hb.test_folder = 'test_primesense'

bopConfig['itodd'] = itodd = DatasetConfig()
itodd.depth_ext = 'tif'
itodd.img_folder = 'gray'
itodd.img_ext = 'tif'
