import os
import cv2
import numpy as np
import mmcv
from .BOPInstanceDataset import BopInstanceDataset, BopInstanceAux
from core.utils.utils import egocentric_to_allocentric
from ..utils.class_id_encoder_decoder import RGB_to_class_id, class_id_to_class_code_images
from lib.pysixd import inout, misc


class GTLoader(BopInstanceAux):
    def __init__(self, copy=False):
        self.copy = copy

    def __call__(self, inst: dict, dataset: BopInstanceDataset) -> dict:
        scene_id, img_id, pose_idx = inst['scene_id'], inst['img_id'], inst['pose_idx']
        fp = dataset.GT_folder / f'{scene_id:06d}' / f'{img_id:06d}_{pose_idx:06d}.png'
        try:
            code = cv2.imread(str(fp), cv2.IMREAD_COLOR)[..., ::-1]
            assert code is not None
            inst['GT'] = code.copy() if self.copy else code
        except:
            inst['GT'] = np.full_like(inst['rgb'], 0)
        return inst

class GT2CodeAux(BopInstanceAux):
    # applied after RandomRotatedMaskCrop, need 'GT_crop' exists
    def __init__(self, key='GT_crop'):
        self.key = key
    def __call__(self, inst: dict, _) -> dict:
        class_id_image = RGB_to_class_id(inst[self.key])
        class_code_images = class_id_to_class_code_images(class_id_image)
        class_code_images = class_code_images.transpose((2, 0, 1))
        inst['code_crop'] = class_code_images
        return inst


class PosePresentationAux(BopInstanceAux):
    def __init__(self, crop_res, R_type='R_allo_6d', t_type='SITE'):
        self.crop_res = crop_res
        self.R_type = R_type
        self.t_type = t_type

    def __call__(self, inst: dict, dataset: BopInstanceDataset) -> dict:
        R = inst['cam_R_obj']
        t = inst['cam_t_obj']

        if self.R_type == 'R_allo_6d':
            allo_pose = egocentric_to_allocentric(np.column_stack((R, t)))
            inst["allo_rot6d"] = allo_pose[:3, :2]
        if self.t_type == 'SITE':
            obj_center = np.matmul(inst['K_crop'], t)
            obj_center = obj_center[:2] / obj_center[2]
            crop_center = (self.crop_res - 1) / 2
            delta_xy = (obj_center - crop_center) / self.crop_res  # [-0.5, 0.5]
            z_ratio = (inst['AABB_crop'][2] - inst['AABB_crop'][0]) / self.crop_res * t[2]
            inst['SITE'] = np.concatenate((delta_xy, z_ratio[np.newaxis]), axis=0)
        return inst

class LoadPointsAux(BopInstanceAux):
    def __init__(self, num_points=3000):
        self.models_path = None
        self.model_points = {}
        self.num_points = num_points

    def init(self, dataset: BopInstanceDataset):
        self.models_path = dataset.dataset_root / dataset.cfg.model_folder

    def __call__(self, inst: dict, dataset: BopInstanceDataset) -> dict:
        obj_id = inst["obj_id"]
        if obj_id not in self.model_points.keys():
            # load model_point to model_points
            model_path = self.models_path / f"obj_{obj_id:06d}.ply"
            model = inout.load_ply(str(model_path))
            assert self.num_points <= model["pts"].shape[0]
            chosen_idx = np.random.choice(model["pts"].shape[0], self.num_points, replace=False)
            self.model_points[obj_id] = model["pts"][chosen_idx, :]
        inst["points"] = self.model_points[obj_id]
        return inst


class LoadSymInfoExtentsAux(BopInstanceAux):
    def __init__(self):
        self.sym_infos = None
        self.extents = None

    def init(self, dataset: BopInstanceDataset):
        models_info_path = dataset.dataset_root / dataset.cfg.model_folder / "models_info.json"
        assert os.path.exists(models_info_path), models_info_path
        models_info = mmcv.load(models_info_path)
        sym_infos = {}
        extents = {}
        for obj_id_str, model_info in models_info.items():
            if "symmetries_discrete" in model_info or "symmetries_continuous" in model_info:
                sym_transforms = misc.get_symmetry_transformations(model_info, max_sym_disc_step=0.01)
                s = np.array([sym["R"] for sym in sym_transforms], dtype=np.float32)
            else:
                s = None
            sym_infos[int(obj_id_str)] = s
            extents[int(obj_id_str)] = np.array([model_info['size_x'], model_info['size_y'], model_info['size_z']],
                                                dtype="float32")
        self.sym_infos = sym_infos
        self.extents = extents
    def __call__(self, inst: dict, dataset: BopInstanceDataset) -> dict:
        obj_id = inst["obj_id"]
        sym_info = self.sym_infos[obj_id]
        inst["sym_info"] = sym_info
        inst["extent"] = self.extents[obj_id]
        return inst
