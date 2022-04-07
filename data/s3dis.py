from .custom import CustomDataset
import torch
import numpy as np
from glob import glob
import os.path as osp
import sys

sys.path.append('../')

from lib.softgroup_ops.functions import softgroup_ops  # noqa


class S3DISDataset(CustomDataset):

    CLASSES = ("ceiling", "floor", "wall", "beam", "column", "window", "door", "chair", "table",
               "bookcase", "sofa", "board", "clutter")

    def get_filenames(self):
        if isinstance(self.prefix, str):
            self.prefix = [self.prefix]
        filenames_all = []
        for p in self.prefix:
            filenames = glob(osp.join(self.data_root, p + '*' + self.suffix))
            assert len(filenames) > 0, f'Empty {p}'
            filenames_all.extend(filenames)
        filenames_all.sort()
        return filenames_all

    def load(self, filename):
        # TODO make file load results consistent
        xyz, rgb, label, instance_label, _, _ = torch.load(filename)
        # subsample data
        if self.training:
            N = xyz.shape[0]
            inds = np.random.choice(N, int(N * 0.25), replace=False)
            xyz = xyz[inds]
            rgb = rgb[inds]
            label = label[inds]
            instance_label = self.getCroppedInstLabel(instance_label, inds)
        return xyz, rgb, label, instance_label

    def crop(self, xyz, step=64):
        xyz_offset = xyz.copy()
        valid_idxs = (xyz_offset.min(1) >= 0) * (
            (xyz < self.voxel_cfg.spatial_shape[1]).sum(1) == 3)

        spatial_shape = np.array([self.voxel_cfg.spatial_shape[1]] * 3)
        room_range = xyz.max(0) - xyz.min(0)
        while (valid_idxs.sum() > self.voxel_cfg.max_npoint):
            step_temp = step
            if valid_idxs.sum() > 1e6:
                step_temp = step * 2
            offset = np.clip(spatial_shape - room_range + 0.001, None, 0) * np.random.rand(3)
            xyz_offset = xyz + offset
            valid_idxs = (xyz_offset.min(1) >= 0) * ((xyz_offset < spatial_shape).sum(1) == 3)
            spatial_shape[:2] -= step_temp

        return xyz_offset, valid_idxs

    def getInstanceInfo(self, xyz, instance_label, label):
        instance_info = np.ones((xyz.shape[0], 9), dtype=np.float32) * -100.0
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
            instance_cls.append(label[cls_loc])
        # assert (0 not in instance_cls) and (1 not in instance_cls)  # sanity check stuff cls

        return instance_num, {
            "instance_info": instance_info,
            "instance_pointnum": instance_pointnum,
            "instance_cls": instance_cls
        }

    def transform_test(self, xyz, rgb, label, instance_label):
        # devide into 4 piecies
        inds = np.arange(xyz.shape[0])
        piece_1 = inds[::4]
        piece_2 = inds[1::4]
        piece_3 = inds[2::4]
        piece_4 = inds[3::4]
        xyz_aug = self.dataAugment(xyz, False, True, True)

        xyz_list = []
        xyz_middle_list = []
        rgb_list = []
        for batch, piece in enumerate([piece_1, piece_2, piece_3, piece_4]):
            xyz_middle = xyz_aug[piece]
            xyz = xyz_middle * self.voxel_cfg.scale
            xyz -= xyz.min(0)
            xyz_list.append(np.concatenate([np.full((xyz.shape[0], 1), batch), xyz], 1))
            xyz_middle_list.append(xyz_middle)
            rgb_list.append(rgb[piece])
        xyz = np.concatenate(xyz_list, 0)
        xyz_middle = np.concatenate(xyz_middle_list, 0)
        rgb = np.concatenate(rgb_list, 0)
        valid_idxs = np.ones(xyz.shape[0], dtype=bool)
        instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)  # TODO remove this
        return xyz, xyz_middle, rgb, label, instance_label

    def collate_fn(self, batch):
        if self.training:
            return super().collate_fn(batch)

        # assume 1 scan only
        (scan_id, loc, loc_float, feat, label, instance_label, inst_num, inst_info, inst_pointnum,
         inst_cls) = batch[0]
        scan_ids = [scan_id]
        locs = loc.long()
        locs_float = loc_float.float()
        feats = feat.float()
        labels = label.long()
        instance_labels = instance_label.long()
        instance_infos = inst_info.float()
        instance_pointnum = torch.tensor([inst_pointnum], dtype=torch.int)
        instance_cls = torch.tensor([inst_cls], dtype=torch.long)
        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.voxel_cfg.spatial_shape[0],
                                None)
        voxel_locs, p2v_map, v2p_map = softgroup_ops.voxelization_idx(locs, 4)
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
            'spatial_shape': spatial_shape
        }
