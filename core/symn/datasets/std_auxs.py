""" surfemb dataset help function """
from typing import Set
from typing import Union

import cv2
import numpy as np
import torch

from .BOPInstanceDataset import BopInstanceDataset, BopInstanceAux
from ..utils.renderer import ObjCoordRenderer

class RgbLoader(BopInstanceAux):
    def __init__(self, copy=False):
        self.copy = copy

    def __call__(self, inst: dict, dataset: BopInstanceDataset) -> dict:
        scene_id, img_id = inst['scene_id'], inst['img_id']
        fp = dataset.data_folder / f'{scene_id:06d}/{dataset.img_folder}/{img_id:06d}.{dataset.img_ext}'
        rgb = cv2.imread(str(fp), cv2.IMREAD_COLOR)[..., ::-1]
        assert rgb is not None
        inst['rgb'] = rgb.copy() if self.copy else rgb
        return inst


class MaskLoader(BopInstanceAux):
    def __init__(self, mask_type=['mask_visib', 'mask']):
        self.mask_type = mask_type

    def __call__(self, inst: dict, dataset: BopInstanceDataset) -> dict:
        scene_id, img_id, pose_idx = inst['scene_id'], inst['img_id'], inst['pose_idx']
        for mask_type in self.mask_type:
            mask_folder = dataset.data_folder / f'{scene_id:06d}' / mask_type
            mask = cv2.imread(str(mask_folder / f'{img_id:06d}_{pose_idx:06d}.png'), cv2.IMREAD_GRAYSCALE)
            assert mask is not None
            mask = mask / 255.
            inst[mask_type] = mask
        return inst

class RandomRotatedMaskCrop(BopInstanceAux):
    def __init__(self, crop_res: int, crop_scale=1.15, max_angle=0, mask_key='mask_visib',
                 crop_keys=('rgb'), crop_keys_crop_res_divide2=('mask', 'mask_visib', 'code'),
                 offset_scale=1., use_bbox_est=False,
                 rgb_interpolation=(cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC)):
        self.crop_res, self.crop_scale = crop_res, crop_scale
        self.max_angle = max_angle
        self.mask_key = mask_key
        self.crop_keys = crop_keys
        self.crop_keys_crop_res_divide2 = crop_keys_crop_res_divide2
        self.rgb_interpolation = rgb_interpolation
        self.offset_scale = offset_scale
        self.use_bbox_est = use_bbox_est
        self.definition_aux = RandomRotatedMaskCropDefinition(self)
        self.apply_aux = RandomRotatedMaskCropApply(self)

    def __call__(self, inst: dict, _) -> dict:
        inst = self.definition_aux(inst, _)
        inst = self.apply_aux(inst, _)
        return inst


class RandomRotatedMaskCropDefinition(BopInstanceAux):
    def __init__(self, parent: RandomRotatedMaskCrop):
        self.p = parent

    def __call__(self, inst: dict, _) -> dict:
        theta = np.random.uniform(-self.p.max_angle, self.p.max_angle)
        S, C = np.sin(theta), np.cos(theta)
        R = np.array((
            (C, -S),
            (S, C),
        ))

        if self.p.use_bbox_est:
            left, top, right, bottom = inst['bbox_est']
        else:
            mask_arg_rotated = np.argwhere(inst[self.p.mask_key])[:, ::-1] @ R.T
            left, top = mask_arg_rotated.min(axis=0)
            right, bottom = mask_arg_rotated.max(axis=0)
        cy, cx = (top + bottom) / 2, (left + right) / 2

        # detector crops can probably be simulated better than this
        size = self.p.crop_res / max(bottom - top, right - left) / self.p.crop_scale
        size = size * np.random.uniform(1 - 0.05 * self.p.offset_scale, 1 + 0.05 * self.p.offset_scale)
        r = self.p.crop_res
        M = np.concatenate((R, [[-cx], [-cy]]), axis=1) * size
        M[:, 2] += r / 2

        offset = (r - r / self.p.crop_scale) / 2 * self.p.offset_scale
        M[:, 2] += np.random.uniform(-offset, offset, 2)
        Ms = np.concatenate((M, [[0, 0, 1]]))

        # calculate axis aligned bounding box in the original image of the rotated crop
        crop_corners = np.array(((0, 0, 1), (0, r, 1), (r, 0, 1), (r, r, 1))) - (0.5, 0.5, 0)  # (4, 3)
        crop_corners = np.linalg.inv(Ms) @ crop_corners.T  # (3, 4)
        crop_corners = crop_corners[:2] / crop_corners[2:]  # (2, 4)
        left, top = np.floor(crop_corners.min(axis=1)).astype(int)
        right, bottom = np.ceil(crop_corners.max(axis=1)).astype(int) + 1
        # left, top = np.maximum((left, top), 0)  # Maybe we don't need this
        right, bottom = np.maximum((right, bottom), (left + 1, top + 1))

        inst['AABB_crop'] = np.array([left, top, right, bottom])
        inst['M_crop'] = M
        inst['K_crop'] = Ms @ inst['K']
        return inst


class RandomRotatedMaskCropApply(BopInstanceAux):
    def __init__(self, parent: RandomRotatedMaskCrop):
        self.p = parent

    def __call__(self, inst: dict, _) -> dict:
        r = self.p.crop_res
        for crop_key in self.p.crop_keys:
            im = inst[crop_key]
            interp = np.random.choice(self.p.rgb_interpolation)
            inst[f'{crop_key}_crop'] = cv2.warpAffine(im, inst['M_crop'], (r, r), flags=interp)
        for crop_key in self.p.crop_keys_crop_res_divide2:
            im = inst[crop_key]
            interp = cv2.INTER_NEAREST if crop_key in ['GT',] else cv2.INTER_LINEAR
            inst[f'{crop_key}_crop'] = cv2.warpAffine(im, inst['M_crop'] * 0.5, (r//2, r//2), flags=interp)
        return inst


class TransformsAux(BopInstanceAux):
    def __init__(self, tfms, key='rgb_crop', crop_key=None):
        self.key = key
        self.tfms = tfms
        self.crop_key = crop_key

    def __call__(self, inst: dict, _) -> dict:
        if self.crop_key is not None:
            left, top, right, bottom = inst[self.crop_key]
            img_slice = slice(top, bottom), slice(left, right)
        else:
            img_slice = slice(None)
        img = inst[self.key]
        img[img_slice] = self.tfms(image=img[img_slice])['image']
        return inst


class KeyFilterOutAux(BopInstanceAux):
    def __init__(self, keys=Set[str]):
        self.keys = keys

    def __call__(self, inst: dict, _) -> dict:
        return {k: v for k, v in inst.items() if k not in self.keys}


class ObjCoordAux(BopInstanceAux):
    def __init__(self, objs, res: int, mask_key='mask_visib_crop', replace_mask=False):
        self.objs, self.res = objs, res
        self.mask_key = mask_key
        self.replace_mask = replace_mask
        self.renderer = None
        self.obj_ids = [k for k in objs.keys()]

    def get_renderer(self):
        # lazy instantiation of renderer to create the context in the worker process
        if self.renderer is None:
            self.renderer = ObjCoordRenderer(self.objs, self.obj_ids, self.res)
        return self.renderer

    def __call__(self, inst: dict, _) -> dict:
        renderer = self.get_renderer()
        K = inst['K_crop']

        obj_coord = renderer.render(inst['obj_id'], K, inst['cam_R_obj'], inst['cam_t_obj']).copy()
        # if self.mask_key is not None:
        #     if self.replace_mask:
        #         mask = obj_coord[..., 3]
        #     else:
        #         mask = obj_coord[..., 3] * inst[self.mask_key] / 255
        #         obj_coord[..., 3] = mask
        #     inst[self.mask_key] = (mask * 255).astype(np.uint8)
        inst['obj_coord'] = obj_coord
        return inst


class NormalizeAux(BopInstanceAux):
    def __init__(self, key='rgb_crop', suffix=''):
        self.key = key
        self.suffix = suffix

    def __call__(self, inst: dict, _) -> dict:
        inst[f'{self.key}{self.suffix}'] = normalize(inst[self.key])
        return inst

def normalize(img: np.ndarray):  # (h, w, 3) -> (3, h, w)
    mu, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    if img.dtype == np.uint8:
        img = img / 255
    img = (img - mu) / std
    return img.transpose(2, 0, 1).astype(np.float32)

def denormalize(img: Union[np.ndarray, torch.Tensor]):
    mu, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    if isinstance(img, torch.Tensor):
        mu, std = [torch.Tensor(v).type(img.dtype).to(img.device)[:, None, None] for v in (mu, std)]
    return img * std + mu