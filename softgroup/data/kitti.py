import os.path as osp
from glob import glob
from pathlib import Path

import numpy as np
import yaml

from .custom import CustomDataset


class KITTIDataset(CustomDataset):

    STUFF = ('road', 'parking', 'sidewalk', 'otherground', 'building', 'fence', 'vegetation',
             'trunk', 'terrain', 'pole', 'traffic-sign')
    THING = ('car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'bicyclist',
             'motorcyclist')
    CLASSES = THING
    NYU_ID = None

    def __init__(self,
                 data_root,
                 prefix,
                 suffix,
                 voxel_cfg=None,
                 training=True,
                 with_label=True,
                 repeat=1,
                 logger=None):
        with open(osp.join(data_root, 'semantic-kitti.yaml'), 'r') as f:
            semkittiyaml = yaml.safe_load(f)
        if prefix == 'train':
            self.split = semkittiyaml['split']['train']
        elif prefix == 'val':
            self.split = semkittiyaml['split']['valid']
        elif prefix == 'test':
            self.split = semkittiyaml['split']['test']
        self.learning_map = semkittiyaml['learning_map']
        self.learning_map_inv = semkittiyaml['learning_map_inv']

        # stuff 0 -> 10, thing 11 -> 18, ignore -100
        for k, v in self.learning_map.items():
            if v == 0:
                new_v = -100
            elif v < 9:
                new_v = v + 10
            else:
                new_v = v - 9
            self.learning_map[k] = new_v
        super(KITTIDataset, self).__init__(data_root, prefix, suffix, voxel_cfg, training,
                                           with_label, repeat, logger)

    def get_filenames(self):
        filenames_all = []
        for p in self.split:
            filenames = glob(
                osp.join(self.data_root, 'sequences', f'{p:02d}', 'velodyne', '*' + self.suffix))
            assert len(filenames) > 0, f'Empty {p}'
            filenames_all.extend(filenames)
        filenames_all = sorted(filenames_all * self.repeat)
        return filenames_all

    def load(self, filename):
        data = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
        xyz, rgb = data[:, :3], data[:, 3:]
        if self.with_label:
            label = np.fromfile(
                filename.replace('velodyne', 'labels').replace('bin', 'label'), dtype=np.int32)
            semantic_label = label & 0xFFFF
            semantic_label = np.vectorize(self.learning_map.__getitem__)(semantic_label)
            stuff_inds = semantic_label <= 10
            instance_label = label
            instance_label[stuff_inds] = -100
        else:
            semantic_label = np.zeros(xyz.shape[0])
            instance_label = np.zeros(xyz.shape[0])
        return xyz, rgb, semantic_label, instance_label

    def getCroppedInstLabel(self, instance_label, valid_idxs):
        instance_label = instance_label[valid_idxs]
        ins_label_map = {}
        new_id = 0
        instance_ids = np.unique(instance_label)
        for id in instance_ids:
            if id == -100:
                ins_label_map[id] = id
                continue
            ins_label_map[id] = new_id
            new_id += 1
        instance_label = np.vectorize(ins_label_map.__getitem__)(instance_label)
        return instance_label

    def transform_train(self, xyz, rgb, semantic_label, instance_label, aug_prob=1.0):
        xyz_middle = self.dataAugment(xyz, True, True, True, True, aug_prob)

        # use smaller scale to speed up elastic
        down = 5
        xyz = xyz_middle * self.voxel_cfg.scale / down
        if np.random.rand() < aug_prob:
            xyz = self.elastic(xyz, 6, 40. / down)
            xyz = self.elastic(xyz, 20, 160. / down)
        xyz = xyz * down

        xyz = xyz - xyz.min(0)
        max_tries = 5
        while (max_tries > 0):
            xyz_offset, valid_idxs = self.crop(xyz)
            if valid_idxs.sum() >= self.voxel_cfg.min_npoint:
                xyz = xyz_offset
                break
            max_tries -= 1
        if valid_idxs.sum() < self.voxel_cfg.min_npoint:
            return None
        xyz = xyz[valid_idxs]
        xyz_middle = xyz_middle[valid_idxs]
        rgb = rgb[valid_idxs]
        semantic_label = semantic_label[valid_idxs]
        instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)
        return xyz, xyz_middle, rgb, semantic_label, instance_label

    def getInstanceInfo(self, xyz, instance_label, semantic_label):
        ret = super().getInstanceInfo(xyz, instance_label, semantic_label)
        instance_num, instance_pointnum, instance_cls, pt_offset_label = ret
        instance_cls = [x - 11 if x != -100 else x for x in instance_cls]
        return instance_num, instance_pointnum, instance_cls, pt_offset_label

    def __getitem__(self, index):
        # add sequence_id to scan_id
        filename = self.filenames[index]
        parts = Path(filename).parts[-4:]
        scan_id = osp.join(*parts).replace(self.suffix, '')
        data = super().__getitem__(index)
        if data is None:
            return data
        return (scan_id, ) + data[1:]
