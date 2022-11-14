""" modified from surfemb BOP detector crop dataset
    this dataset is used for training and debug for test dataset,
    if you want to test, use BOPDetCropDataset which uses detection result.
"""
import json
from pathlib import Path
from typing import Sequence
from collections import defaultdict
import numpy as np
from tqdm import tqdm

import torch.utils.data

from core.symn.utils.BOPConfig import DatasetConfig
from core.symn.datasets.BOPInstanceDataset import BopInstanceAux


class BOPDetCropDataset(torch.utils.data.Dataset):
    def __init__(
            self, dataset_root: Path, detections, cfg: DatasetConfig,
            auxs: Sequence[BopInstanceAux],
            obj_ids, score_thr=0.0, top_k_per_obj=1,
            show_progressbar=True,
    ):
        self.data_folder = dataset_root / cfg.test_folder
        self.img_folder = cfg.img_folder
        self.depth_folder = cfg.depth_folder
        self.img_ext = cfg.img_ext
        self.depth_ext = cfg.depth_ext
        self.auxs = auxs
        obj_idxs = {obj_id: idx for idx, obj_id in enumerate(obj_ids)}
        self.instances = []
        scene_ids = sorted([int(p.name) for p in self.data_folder.glob('*')])
        for scene_id in tqdm(scene_ids, 'loading crop info') if show_progressbar else scene_ids:
            scene_folder = self.data_folder / f'{scene_id:06d}'
            scene_camera = json.load((scene_folder / 'scene_camera.json').open())
            for img_id, camera in scene_camera.items():
                K = np.array(camera['cam_K']).reshape((3, 3)).copy()
                scene_im_id = str(scene_id) + '/' + str(img_id)
                dets = detections.get(scene_im_id, None)
                if dets is None:
                    continue
                obj_annotations = {obj_id: [] for obj_id in obj_ids}
                for det in dets:
                    obj_id = det['obj_id']
                    det_score = det['det_score']
                    bbox_est = det['bbox_est']
                    det_time = det['det_time']
                    if obj_id not in obj_ids:
                        continue
                    if det_score < score_thr:
                        continue
                    inst = dict(scene_id=scene_id, img_id=int(img_id), K=K, obj_id=obj_id,
                                bbox_est=bbox_est, det_time=det_time, det_score=det_score,
                                obj_idx=obj_idxs[obj_id])
                    obj_annotations[obj_id].append(inst)
                for obj_id, annotations in obj_annotations.items():
                    scores = [ann["det_score"] for ann in annotations]
                    sel_annos = [ann for _, ann in sorted(zip(scores, annotations),
                                                          key=lambda pair: pair[0], reverse=True)][:top_k_per_obj]
                    self.instances.extend(sel_annos)

        for aux in self.auxs:
            aux.init(self)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i):
        instance = self.instances[i].copy()
        for aux in self.auxs:
            instance = aux(instance, self)
        return instance

if __name__ == '__main__':
    from core.symn.utils.BOPConfig import tless
    data = BOPDetCropDataset(dataset_root=Path('/path/to/tless'), cfg=tless, obj_ids=range(1, 31),
                             detection_files=Path(f'data/detection_results/tless'))
    print(len(data))
