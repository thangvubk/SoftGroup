import os.path as osp
from glob import glob

import numpy as np
import torch

from ..ops import voxelization_idx
from .custom import CustomDataset


class S3DISDataset(CustomDataset):

    CLASSES = ('ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'chair', 'table',
               'bookcase', 'sofa', 'board', 'clutter')

    def __init__(self, x4_split=False, **kwargs):
        super().__init__(**kwargs)
        self.x4_split = x4_split

    def get_filenames(self):
        if isinstance(self.prefix, str):
            self.prefix = [self.prefix]
        filenames_all = []
        for p in self.prefix:
            filenames = glob(osp.join(self.data_root, p + '*' + self.suffix))
            assert len(filenames) > 0, f'Empty {p}'
            filenames_all.extend(filenames)
        filenames_all = sorted(filenames_all * self.repeat)
        return filenames_all

    def load(self, filename):
        xyz, rgb, semantic_label, instance_label, _, _ = torch.load(filename)
        # subsample data
        if self.training and self.x4_split:
            N = xyz.shape[0]
            inds = np.random.choice(N, int(N * 0.25), replace=False)
            xyz = xyz[inds]
            rgb = rgb[inds]
            semantic_label = semantic_label[inds]
            instance_label = self.getCroppedInstLabel(instance_label, inds)
        return xyz, rgb, semantic_label, instance_label

    def crop(self, xyz, step=64):
        return super().crop(xyz, step=step)

    def transform_test(self, xyz, rgb, semantic_label, instance_label):
        if not self.x4_split:
            return super().transform_test(xyz, rgb, semantic_label, instance_label)
        # devide into 4 piecies
        inds = np.arange(xyz.shape[0])
        piece_1 = inds[::4]
        piece_2 = inds[1::4]
        piece_3 = inds[2::4]
        piece_4 = inds[3::4]
        xyz_aug = self.dataAugment(xyz, False, False, False)

        xyz_list = []
        xyz_middle_list = []
        rgb_list = []
        semantic_label_list = []
        instance_label_list = []
        for batch, piece in enumerate([piece_1, piece_2, piece_3, piece_4]):
            xyz_middle = xyz_aug[piece]
            xyz = xyz_middle * self.voxel_cfg.scale
            xyz -= xyz.min(0)
            xyz_list.append(np.concatenate([np.full((xyz.shape[0], 1), batch), xyz], 1))
            xyz_middle_list.append(xyz_middle)
            rgb_list.append(rgb[piece])
            semantic_label_list.append(semantic_label[piece])
            instance_label_list.append(instance_label[piece])
        xyz = np.concatenate(xyz_list, 0)
        xyz_middle = np.concatenate(xyz_middle_list, 0)
        rgb = np.concatenate(rgb_list, 0)
        semantic_label = np.concatenate(semantic_label_list, 0)
        instance_label = np.concatenate(instance_label_list, 0)
        valid_idxs = np.ones(xyz.shape[0], dtype=bool)
        instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)  # TODO remove this
        return xyz, xyz_middle, rgb, semantic_label, instance_label

    def collate_fn(self, batch):
        if self.training or not self.x4_split:
            return super().collate_fn(batch)

        # assume 1 scan only
        (scan_id, coord, coord_float, feat, semantic_label, instance_label, inst_num, inst_pointnum,
         inst_cls, pt_offset_label) = batch[0]
        scan_ids = [scan_id]
        coords = coord.long()
        batch_idxs = torch.zeros_like(coord[:, 0].int())
        coords_float = coord_float.float()
        feats = feat.float()
        semantic_labels = semantic_label.long()
        instance_labels = instance_label.long()
        instance_pointnum = torch.tensor([inst_pointnum], dtype=torch.int)
        instance_cls = torch.tensor([inst_cls], dtype=torch.long)
        pt_offset_labels = pt_offset_label.float()
        spatial_shape = np.clip((coords.max(0)[0][1:] + 1).numpy(), self.voxel_cfg.spatial_shape[0],
                                None)
        voxel_coords, v2p_map, p2v_map = voxelization_idx(coords, 4)
        return {
            'scan_ids': scan_ids,
            'batch_idxs': batch_idxs,
            'voxel_coords': voxel_coords,
            'p2v_map': p2v_map,
            'v2p_map': v2p_map,
            'coords_float': coords_float,
            'feats': feats,
            'semantic_labels': semantic_labels,
            'instance_labels': instance_labels,
            'instance_pointnum': instance_pointnum,
            'instance_cls': instance_cls,
            'pt_offset_labels': pt_offset_labels,
            'spatial_shape': spatial_shape,
            'batch_size': 4
        }
