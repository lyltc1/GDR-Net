import torch
from torch.utils.data.sampler import BatchSampler
from torch.utils.data import DataLoader
from torch import set_num_threads
from torch.utils.data.dataset import ConcatDataset
from core.symn.utils.BOPConfig import root, bopConfig
from core.symn.datasets.get_detection_results import *
from ..utils.obj import load_objs
from .BOPInstanceDataset import BopInstanceDataset
from .BOPDetCropDataset import BOPDetCropDataset
from .std_auxs import RgbLoader, MaskLoader, RandomRotatedMaskCrop, ObjCoordAux, NormalizeAux, KeyFilterOutAux
from .GDRN_aux import AugRgbAux, ReplaceBgAux
from .symn_aux import GTLoader, GT2CodeAux, PosePresentationAux, LoadSymInfoExtentsAux, LoadPointsAux
from core.utils.my_distributed_sampler import TrainingSampler, InferenceSampler
from core.utils.my_comm import comm
from core.utils.dataset_utils import trivial_batch_collator

import cv2


def get_aux(cfg, test, pbr, debug=False):
    dataset_name = cfg.DATASETS.NAME
    res_crop = cfg.DATASETS.RES_CROP
    bop_config = bopConfig[dataset_name]
    obj_ids = cfg.DATASETS.OBJ_IDS
    objs, obj_ids = load_objs(bop_config, obj_ids)

    auxs = [RgbLoader()]
    if test is False:
        # train_aux
        auxs.extend([MaskLoader(),
                     GTLoader(),
                     RandomRotatedMaskCrop(res_crop, mask_key='mask_visib',
                                           crop_keys=('rgb',),
                                           crop_keys_crop_res_divide2=('mask', 'mask_visib', 'GT'),
                                           rgb_interpolation=cv2.INTER_LINEAR),
                     GT2CodeAux(),
                     PosePresentationAux(res_crop, R_type='R_allo_6d', t_type='SITE'),
                     LoadPointsAux(num_points=cfg.MODEL.PNP_NET.PM_NUM_POINTS),
                     LoadSymInfoExtentsAux(),
                     ])
        if pbr is False:
            # train_real_aux
            auxs.extend([ReplaceBgAux(cfg.DATASETS.BG_AUG_PROB, cfg.DATASETS.BG_AUG_TYPE),
                         AugRgbAux(cfg.DATASETS.COLOR_AUG_PROB),
                         ])
    else:
        # test_aux
        auxs.extend([RandomRotatedMaskCrop(res_crop, mask_key='mask_visib',
                                           crop_keys=('rgb',),
                                           crop_keys_crop_res_divide2=tuple(),
                                           rgb_interpolation=cv2.INTER_LINEAR),
                     ])
    auxs.extend([NormalizeAux(), ])
    if debug is False:
        auxs.extend([KeyFilterOutAux({'scene_id', 'img_id', 'rgb', 'mask', 'mask_visib',
                                      'GT', 'GT_crop', 'K', 'bbox_obj', 'bbox_visib'})])
    else:
        auxs.extend([ObjCoordAux(objs, res_crop),])
    return auxs

def build_train_BOP_instance_dataset(cfg, dataset_type, debug=False):
    """ dataset_type is cfg.DATASETS.TRAIN or cfg.DATASETS.TRAIN2 """
    dataset_name = cfg.DATASETS.NAME
    res_crop = cfg.DATASETS.RES_CROP
    bop_config = bopConfig[dataset_name]
    bop_config.dataset = dataset_name
    dataset_root = root / "datasets/BOP_DATASETS" / dataset_name
    obj_ids = cfg.DATASETS.OBJ_IDS
    objs, obj_ids = load_objs(bop_config, obj_ids)
    if isinstance(dataset_type, str):
        dataset_type = [dataset_type]
    assert len(dataset_type), dataset_type
    if len(dataset_type):
        dataset = []
    else:
        dataset = None
    for name in dataset_type:
        if name == "real":
            # we use both ReplaceBgAux and AugRgbAux in BOP real dataset
            auxs_real = get_aux(cfg, test=False, pbr=False, debug=debug)
            real_dataset = BopInstanceDataset(dataset_root, pbr=False, test=False, cfg=bop_config, obj_ids=obj_ids,
                                              auxs=auxs_real)
            dataset.append(real_dataset)
        elif name == "pbr":
            # we do not replace background in BOP pbr dataset
            auxs_pbr = get_aux(cfg, test=False, pbr=True, debug=debug)
            pbr_dataset = BopInstanceDataset(dataset_root, pbr=True, test=False, cfg=bop_config, obj_ids=obj_ids,
                                             auxs=auxs_pbr)
            dataset.append(pbr_dataset)
        else:
            raise NotImplementedError
    if len(dataset) > 1:
        dataset = ConcatDataset(dataset)
    elif len(dataset) == 1:
        dataset = dataset[0]
    return dataset


def build_test_BOP_instance_dataset(cfg, debug=False):
    """ build dataset in cfg.DATASETS.TEST, for debug only """
    dataset_name = cfg.DATASETS.NAME
    res_crop = cfg.DATASETS.RES_CROP
    bop_config = bopConfig[dataset_name]
    bop_config.dataset = dataset_name
    dataset_root = root / "datasets/BOP_DATASETS" / dataset_name
    obj_ids = cfg.DATASETS.OBJ_IDS
    objs, obj_ids = load_objs(bop_config, obj_ids)
    # build test_dataset
    test_dataset = None
    if cfg.DATASETS.TEST == "test":
        auxs_test = get_aux(cfg, test=True, pbr=False, debug=debug)
        test_dataset = BopInstanceDataset(dataset_root, pbr=False, test=True, cfg=bop_config, obj_ids=obj_ids,
                                          auxs=auxs_test)
    else:
        raise NotImplementedError
    return test_dataset


def build_BOP_det_crop_dataset(cfg):
    """ build dataset in cfg.DATASETS.TEST """
    dataset_name = cfg.DATASETS.NAME
    res_crop = cfg.DATASETS.RES_CROP
    bop_config = bopConfig[dataset_name]
    bop_config.dataset = dataset_name
    dataset_root = root / "datasets/BOP_DATASETS" / dataset_name
    obj_ids = cfg.DATASETS.OBJ_IDS
    objs, obj_ids = load_objs(bop_config, obj_ids)
    # build test_dataset
    if cfg.DATASETS.TEST == "test":
        auxs_test = [RgbLoader(),
                     RandomRotatedMaskCrop(res_crop, crop_keys=('rgb',), use_bbox_est=True,
                                           rgb_interpolation=cv2.INTER_LINEAR),
                     NormalizeAux(),
                     KeyFilterOutAux({'rgb',})]
        detections = eval("get_detection_results_" + cfg.DATASETS.TEST_DETECTION_TYPE)(cfg.DATASETS.TEST_DETECTION_PATH)
        test_dataset = BOPDetCropDataset(dataset_root, detections, cfg=bop_config, auxs=auxs_test, obj_ids=obj_ids,
                                          score_thr=cfg.DATASETS.TEST_SCORE_THR, top_k_per_obj=cfg.DATASETS.TEST_TOP_K_PER_OBJ)
    else:
        raise NotImplementedError
    return test_dataset

def build_train_dataloader(cfg, dataset):
    """ Build a batched dataloader for training. """
    world_size = comm.get_world_size()
    total_batch_size = cfg.TRAIN.TOTAL_BATCH_SIZE
    batch_size = cfg.TRAIN.BATCH_SIZE
    num_workers = cfg.TRAIN.NUM_WORKERS
    assert (
            total_batch_size and total_batch_size == world_size * batch_size
    ), "check cfg.TRAIN.TOTAL_BATCH_SIZE and cfg.TRAIN.BATCH_SIZE"
    if num_workers > 0:
        set_num_threads(num_workers)
    sampler = TrainingSampler(len(dataset))
    batch_sampler = BatchSampler(sampler, batch_size, drop_last=True)
    data_loader = DataLoader(dataset, batch_sampler=batch_sampler,
                             num_workers=num_workers, collate_fn=trivial_batch_collator)
    return data_loader


def build_test_dataloader(cfg, dataset):
    """ Build a batched dataloader for testing. """
    world_size = comm.get_world_size()
    total_batch_size = cfg.TRAIN.TOTAL_BATCH_SIZE
    batch_size = cfg.TRAIN.BATCH_SIZE
    num_workers = cfg.TRAIN.NUM_WORKERS
    assert (
            total_batch_size and total_batch_size == world_size * batch_size
    ), "check cfg.TRAIN.TOTAL_BATCH_SIZE and cfg.TRAIN.BATCH_SIZE"
    if num_workers > 0:
        set_num_threads(num_workers)
    sampler = InferenceSampler(len(dataset))
    batch_sampler = BatchSampler(sampler, 1, drop_last=False)
    data_loader = DataLoader(dataset, batch_sampler=batch_sampler,
                             num_workers=num_workers, collate_fn=trivial_batch_collator)
    return data_loader

def batch_data_train(data, device="cuda"):
    batch = {}
    for key in ["obj_id", "obj_idx"]:
        batch[key] = torch.tensor([d[key] for d in data], dtype=torch.long).to(device, non_blocking=True)
    for key in ["rgb_crop", "mask_visib_crop", "code_crop", "K_crop", "extent",
                "cam_R_obj", "cam_t_obj", "allo_rot6d", "SITE", "points"]:
        batch[key] = torch.stack([torch.tensor(d[key], dtype=torch.float32) for d in data], dim=0).to(device, non_blocking=True)
    for key in ["AABB_crop"]:
        batch[key] = torch.stack([torch.tensor(d[key], dtype=torch.long) for d in data], dim=0).to(device, non_blocking=True)
    for key in ["sym_info"]:
        batch[key] = [d[key] for d in data]
    return batch

def batch_data_test(data, device="cuda"):
    batch = {}
    for key in ["obj_id", "obj_idx"]:
        batch[key] = torch.tensor([d[key] for d in data], dtype=torch.long).to(device, non_blocking=True)
    for key in ["rgb_crop", "K_crop", "points"]:
        batch[key] = torch.stack([torch.tensor(d[key], dtype=torch.float32) for d in data], dim=0).to(device, non_blocking=True)
    for key in ["AABB_crop"]:
        batch[key] = torch.stack([torch.tensor(d[key], dtype=torch.long) for d in data], dim=0).to(device, non_blocking=True)
    for key in ["sym_info"]:
        batch[key] = [d[key] for d in data]
    return batch
