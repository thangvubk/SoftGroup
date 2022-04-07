import math
import numpy as np
import os.path as osp
import scipy.interpolate
import scipy.ndimage
import sys
import torch
from glob import glob
from torch.utils.data import Dataset

sys.path.append('../')

from lib.softgroup_ops.functions import softgroup_ops  # noqa


class CustomDataset(Dataset):

    CLASSES = None

    def __init__(self, data_root, prefix, suffix, voxel_cfg=None, training=True, repeat=1, logger=None):
        self.data_root = data_root
        self.prefix = prefix
        self.suffix = suffix
        self.voxel_cfg = voxel_cfg
        self.training = training
        self.repeat = repeat
        self.logger = logger
        self.filenames = self.get_filenames()

    def get_filenames(self):
        filenames = glob(osp.join(self.data_root, self.prefix, '*' + self.suffix))
        assert len(filenames) > 0, 'Empty dataset.'
        filenames = sorted(filenames * self.repeat)
        self.logger.info(f'Load dataset: {len(filenames)} scans')

    def load(self, filename):
        return torch.load(filename)

    def __len__(self):
        return len(self.filenames)

    def elastic(self, x, gran, mag):
        blur0 = np.ones((3, 1, 1)).astype('float32') / 3
        blur1 = np.ones((1, 3, 1)).astype('float32') / 3
        blur2 = np.ones((1, 1, 3)).astype('float32') / 3

        bb = np.abs(x).max(0).astype(np.int32) // gran + 3
        noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in bb]
        interp = [
            scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0)
            for n in noise
        ]

        def g(x_):
            return np.hstack([i(x_)[:, None] for i in interp])

        return x + g(x) * mag

    def getInstanceInfo(self, xyz, instance_label, label):
        '''
        :param xyz: (n, 3)
        :param instance_label: (n), int, (0~nInst-1, -100)
        :return: instance_num, dict
        '''
        instance_info = np.ones(
            (xyz.shape[0], 9), dtype=np.float32
        ) * -100.0  # (n, 9), float, (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
        instance_pointnum = []  # (nInst), int
        instance_cls = []
        instance_num = int(instance_label.max()) + 1
        for i_ in range(instance_num):
            inst_idx_i = np.where(instance_label == i_)

            # instance_info
            xyz_i = xyz[inst_idx_i]
            min_xyz_i = xyz_i.min(0)
            max_xyz_i = xyz_i.max(0)
            mean_xyz_i = xyz_i.mean(0)
            instance_info_i = instance_info[inst_idx_i]
            instance_info_i[:, 0:3] = mean_xyz_i
            instance_info_i[:, 3:6] = min_xyz_i
            instance_info_i[:, 6:9] = max_xyz_i
            instance_info[inst_idx_i] = instance_info_i

            # instance_pointnum
            instance_pointnum.append(inst_idx_i[0].size)
            cls_loc = inst_idx_i[0][0]

            # ignore 2 first classes (floor, ceil)
            cls = label[cls_loc] - 2 if label[cls_loc] != -100 else label[cls_loc]
            instance_cls.append(cls)
        return instance_num, {
            "instance_info": instance_info,
            "instance_pointnum": instance_pointnum,
            "instance_cls": instance_cls
        }

    def dataAugment(self, xyz, jitter=False, flip=False, rot=False):
        m = np.eye(3)
        if jitter:
            m += np.random.randn(3, 3) * 0.1
        if flip:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1  # flip x randomly
        if rot:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0],
                              [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])  # rotation
        return np.matmul(xyz, m)

    def crop(self, xyz):
        '''
        :param xyz: (n, 3) >= 0
        '''
        xyz_offset = xyz.copy()
        valid_idxs = (xyz_offset.min(1) >= 0)
        assert valid_idxs.sum() == xyz.shape[0]

        spatial_shape = np.array([self.voxel_cfg.spatial_shape[1]] * 3)
        room_range = xyz.max(0) - xyz.min(0)
        while (valid_idxs.sum() > self.voxel_cfg.max_npoint):
            offset = np.clip(spatial_shape - room_range + 0.001, None, 0) * np.random.rand(3)
            xyz_offset = xyz + offset
            valid_idxs = (xyz_offset.min(1) >= 0) * ((xyz_offset < spatial_shape).sum(1) == 3)
            spatial_shape[:2] -= 32

        return xyz_offset, valid_idxs

    def getCroppedInstLabel(self, instance_label, valid_idxs):
        instance_label = instance_label[valid_idxs]
        j = 0
        while (j < instance_label.max()):
            if (len(np.where(instance_label == j)[0]) == 0):
                instance_label[instance_label == instance_label.max()] = j
            j += 1
        return instance_label

    def transform_train(self, xyz, rgb, label, instance_label):
        xyz_middle = self.dataAugment(xyz, True, True, True)
        xyz = xyz_middle * self.voxel_cfg.scale
        xyz = self.elastic(xyz, 6 * self.voxel_cfg.scale // 50, 40 * self.voxel_cfg.scale / 50)
        xyz = self.elastic(xyz, 20 * self.voxel_cfg.scale // 50, 160 * self.voxel_cfg.scale / 50)
        xyz -= xyz.min(0)
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
        label = label[valid_idxs]
        instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)
        return xyz, xyz_middle, rgb, label, instance_label

    def transform_test(self, xyz, rgb, label, instance_label):
        xyz_middle = self.dataAugment(xyz, False, True, True)
        xyz = xyz_middle * self.voxel_cfg.scale
        xyz -= xyz.min(0)
        valid_idxs = np.ones(xyz.shape[0], dtype=bool)
        instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)  # TODO remove this
        return xyz, xyz_middle, rgb, label, instance_label

    def __getitem__(self, index):
        filename = self.filenames[index]
        scan_id = osp.basename(filename).replace(self.suffix, '')
        data = self.load(filename)
        data = self.transform_train(*data) if self.training else self.transform_test(*data)
        if data is None:
            return None
        xyz, xyz_middle, rgb, label, instance_label = data
        inst_num, inst_infos = self.getInstanceInfo(xyz_middle, instance_label.astype(np.int32),
                                                    label)
        inst_info = inst_infos["instance_info"]
        inst_pointnum = inst_infos["instance_pointnum"]
        inst_cls = inst_infos["instance_cls"]
        loc = torch.from_numpy(xyz).long()
        loc_float = torch.from_numpy(xyz_middle)
        feat = torch.from_numpy(rgb).float()
        if self.training:
            feat += torch.randn(3) * 0.1
        label = torch.from_numpy(label)
        instance_label = torch.from_numpy(instance_label)
        inst_info = torch.from_numpy(inst_info)
        return (scan_id, loc, loc_float, feat, label, instance_label, inst_num, inst_info,
                inst_pointnum, inst_cls)

    def collate_fn(self, batch):
        scan_ids = []
        locs = []
        locs_float = []
        feats = []
        labels = []
        instance_labels = []

        instance_infos = []  # (N, 9)
        instance_pointnum = []  # (total_nInst), int
        instance_cls = []  # (total_nInst), long

        batch_offsets = [0]

        total_inst_num = 0
        batch_id = 0
        for data in batch:
            if data is None:
                continue
            (scan_id, loc, loc_float, feat, label, instance_label, inst_num, inst_info,
             inst_pointnum, inst_cls) = data
            instance_label[np.where(instance_label != -100)] += total_inst_num
            total_inst_num += inst_num
            batch_offsets.append(batch_offsets[-1] + loc.size(0))
            scan_ids.append(scan_id)
            locs.append(torch.cat([loc.new_full((loc.size(0), 1), batch_id), loc], 1))
            locs_float.append(loc_float)
            feats.append(feat)
            labels.append(label)
            instance_labels.append(instance_label)
            instance_infos.append(inst_info)
            instance_pointnum.extend(inst_pointnum)
            instance_cls.extend(inst_cls)
            batch_id += 1
        assert batch_id > 0, 'empty batch'
        if batch_id < len(batch):
            self.logger.info(f'batch is truncated from size {len(batch)} to {batch_id}')

        # merge all the scenes in the batch
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)

        locs = torch.cat(locs, 0)  # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        locs_float = torch.cat(locs_float, 0).to(torch.float32)  # float (N, 3)
        feats = torch.cat(feats, 0)  # float (N, C)
        labels = torch.cat(labels, 0).long()  # long (N)
        instance_labels = torch.cat(instance_labels, 0).long()  # long (N)
        instance_infos = torch.cat(instance_infos,
                                   0).to(torch.float32)  # float (N, 9) (meanxyz, minxyz, maxxyz)
        instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)  # int (total_nInst)
        instance_cls = torch.tensor(instance_cls, dtype=torch.long)  # long (total_nInst)

        spatial_shape = np.clip(
            locs.max(0)[0][1:].numpy() + 1, self.voxel_cfg.spatial_shape[0], None)
        voxel_locs, p2v_map, v2p_map = softgroup_ops.voxelization_idx(locs, 1)
        return {
            'scan_ids': scan_ids,
            'locs': locs,
            'voxel_locs': voxel_locs,
            'p2v_map': p2v_map,
            'v2p_map': v2p_map,
            'locs_float': locs_float,
            'feats': feats,
            'labels': labels,
            'instance_labels': instance_labels,
            'instance_info': instance_infos,
            'instance_pointnum': instance_pointnum,
            'instance_cls': instance_cls,
            'offsets': batch_offsets,
            'spatial_shape': spatial_shape,
            'batch_size': batch_id,
        }
