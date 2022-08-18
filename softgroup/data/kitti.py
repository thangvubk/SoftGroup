import os.path as osp
from glob import glob

import numpy as np
import yaml

from .custom import CustomDataset


class KITTIDataset(CustomDataset):

    STUFF = ('road', 'parking', 'sidewalk', 'otherground', 'building', 'fence', 'vegetation',
             'trunk', 'terrain', 'pole', 'traffic-sign')
    THING = ('car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'bicyclist',
             'motorcyclist')
    CLASSES = None
    NYU_ID = None

    def __init__(self,
                 data_root,
                 prefix,
                 suffix,
                 voxel_cfg=None,
                 training=True,
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

        # stuff 0 -> 10, thing 11 -> 18, ignore -100
        for k, v in self.learning_map.items():
            if v == 0:
                new_v = -100
            elif v < 9:
                new_v = v + 10
            else:
                new_v = v - 9
            self.learning_map[k] = new_v
        super(KITTIDataset, self).__init__(data_root, prefix, suffix, voxel_cfg, training, repeat,
                                           logger)

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
        label = np.fromfile(
            filename.replace('velodyne', 'labels').replace('bin', 'label'), dtype=np.uint32)
        # semantic and instance laels are stored in 16 low-high bits
        semantic_label = label & 0xFFFF
        semantic_label = np.vectorize(self.learning_map.__getitem__)(semantic_label)
        stuff_inds = semantic_label <= 10
        instance_label = label
        instance_label[stuff_inds] = -100
        instance_label[instance_label == 0] = -100
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
