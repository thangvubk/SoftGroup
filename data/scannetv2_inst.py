import os, sys, glob, math, numpy as np
import scipy.ndimage
import scipy.interpolate
import torch
from torch.utils.data import DataLoader

sys.path.append('../')

from util.config import cfg
from util.log import logger
from lib.softgroup_ops.functions import softgroup_ops

import torch.distributed as dist

class Dataset:
    def __init__(self, test=False):
        self.data_root = cfg.data_root
        self.dataset = cfg.dataset
        self.filename_suffix = cfg.filename_suffix

        self.batch_size = cfg.batch_size
        self.train_workers = cfg.train_workers
        self.val_workers = cfg.train_workers

        self.full_scale = cfg.full_scale
        self.scale = cfg.scale
        self.max_npoint = cfg.max_npoint
        self.mode = cfg.mode

        self.train_split = getattr(cfg, 'train_split', 'train')

        if test:
            self.test_split = cfg.split  # val or test
            self.test_workers = cfg.test_workers
            cfg.batch_size = 1


    def trainLoader(self):
        if self.train_split == 'trainval':
            train_file_names = sorted(glob.glob(os.path.join(self.data_root, self.dataset, 'train', '*' + self.filename_suffix))
                + glob.glob(os.path.join(self.data_root, self.dataset, 'val', '*' + self.filename_suffix))
            )
        elif self.train_split == 'train':
            train_file_names = sorted(glob.glob(os.path.join(self.data_root, self.dataset, 'train', '*' + self.filename_suffix)))
        else:
            raise Exception

        self.train_files = [torch.load(i) for i in train_file_names]

        logger.info('Training samples: {}'.format(len(self.train_files)))

        train_set = list(range(len(self.train_files)))
        self.train_data_loader = DataLoader(train_set, batch_size=self.batch_size, collate_fn=self.trainMerge, num_workers=self.train_workers,
                                            shuffle=True, sampler=None, drop_last=True, pin_memory=True)
        

    def dist_trainLoader(self):
        train_file_names = sorted(glob.glob(os.path.join(self.data_root, self.dataset, 'train', '*' + self.filename_suffix)))
        self.train_files = [torch.load(i) for i in train_file_names]

        logger.info('Training samples: {}'.format(len(self.train_files)))

        train_set = list(range(len(self.train_files)))
        # self.train_data_loader = DataLoader(train_set, batch_size=self.batch_size, collate_fn=self.trainMerge, num_workers=self.train_workers,
        #                                     shuffle=True, sampler=None, drop_last=True, pin_memory=True)
        
        # world_size = dist.get_world_size()
        # rank = dist.get_rank()
        # self.data_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=world_size, rank=rank)
        self.data_sampler = torch.utils.data.distributed.DistributedSampler(train_set)

        self.train_data_loader = DataLoader(train_set, batch_size=self.batch_size, 
                                    collate_fn=self.trainMerge, 
                                    num_workers=self.train_workers,
                                    shuffle=False, sampler=self.data_sampler, 
                                    drop_last=False, pin_memory=True)
        


    def valLoader(self):
        val_file_names = sorted(glob.glob(os.path.join(self.data_root, self.dataset, 'val', '*' + self.filename_suffix)))
        self.val_files = [torch.load(i) for i in val_file_names]

        logger.info('Validation samples: {}'.format(len(self.val_files)))

        val_set = list(range(len(self.val_files)))
        self.val_data_loader = DataLoader(val_set, batch_size=self.batch_size, collate_fn=self.valMerge, num_workers=self.val_workers,
                                          shuffle=False, drop_last=False, pin_memory=True)


    def testLoader(self):
        self.test_file_names = sorted(glob.glob(os.path.join(self.data_root, self.dataset, self.test_split, '*' + self.filename_suffix)))
        self.test_files = [torch.load(i) for i in self.test_file_names]

        logger.info('Testing samples ({}): {}'.format(self.test_split, len(self.test_files)))

        test_set = list(np.arange(len(self.test_files)))
        self.test_data_loader = DataLoader(test_set, batch_size=1, collate_fn=self.testMerge, num_workers=self.test_workers,
                                           shuffle=False, drop_last=False, pin_memory=True)

    # Elastic distortion
    def elastic(self, x, gran, mag):
        blur0 = np.ones((3, 1, 1)).astype('float32') / 3
        blur1 = np.ones((1, 3, 1)).astype('float32') / 3
        blur2 = np.ones((1, 1, 3)).astype('float32') / 3

        bb = np.abs(x).max(0).astype(np.int32)//gran + 3
        noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        ax = [np.linspace(-(b-1)*gran, (b-1)*gran, b) for b in bb]
        interp = [scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]
        def g(x_):
            return np.hstack([i(x_)[:,None] for i in interp])
        return x + g(x) * mag


    def getInstanceInfo(self, xyz, instance_label, label):
        '''
        :param xyz: (n, 3)
        :param instance_label: (n), int, (0~nInst-1, -100)
        :return: instance_num, dict
        '''
        instance_info = np.ones((xyz.shape[0], 9), dtype=np.float32) * -100.0   # (n, 9), float, (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
        instance_pointnum = []   # (nInst), int
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
        assert (0 not in instance_cls) and (1 not in instance_cls)  # sanity check stuff cls

        return instance_num, {"instance_info": instance_info, "instance_pointnum": instance_pointnum,
                              "instance_cls": instance_cls}


    def dataAugment(self, xyz, jitter=False, flip=False, rot=False):
        m = np.eye(3)
        if jitter:
            m += np.random.randn(3, 3) * 0.1
        if flip:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1  # flip x randomly
        if rot:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])  # rotation
        return np.matmul(xyz, m)


    def crop(self, xyz):
        '''
        :param xyz: (n, 3) >= 0
        '''
        xyz_offset = xyz.copy()
        valid_idxs = (xyz_offset.min(1) >= 0)
        assert valid_idxs.sum() == xyz.shape[0]

        full_scale = np.array([self.full_scale[1]] * 3)
        room_range = xyz.max(0) - xyz.min(0)
        while (valid_idxs.sum() > self.max_npoint):
            offset = np.clip(full_scale - room_range + 0.001, None, 0) * np.random.rand(3)
            xyz_offset = xyz + offset
            valid_idxs = (xyz_offset.min(1) >= 0) * ((xyz_offset < full_scale).sum(1) == 3)
            full_scale[:2] -= 32

        return xyz_offset, valid_idxs


    def getCroppedInstLabel(self, instance_label, valid_idxs):
        instance_label = instance_label[valid_idxs]
        j = 0
        while (j < instance_label.max()):
            if (len(np.where(instance_label == j)[0]) == 0):
                instance_label[instance_label == instance_label.max()] = j
            j += 1
        return instance_label


    def trainMerge(self, id):
        locs = []
        locs_float = []
        feats = []
        labels = []
        instance_labels = []

        instance_infos = []  # (N, 9)
        instance_pointnum = []  # (total_nInst), int
        instance_cls = [] # (total_nInst), long

        batch_offsets = [0]

        total_inst_num = 0
        for i, idx in enumerate(id):
            xyz_origin, rgb, label, instance_label = self.train_files[idx]


            # jitter / flip x / rotation
            xyz_middle = self.dataAugment(xyz_origin, True, True, True)

            # scale
            xyz = xyz_middle * self.scale

            # elastic
            xyz = self.elastic(xyz, 6 * self.scale // 50, 40 * self.scale / 50)
            xyz = self.elastic(xyz, 20 * self.scale // 50, 160 * self.scale / 50)

            # offset
            xyz -= xyz.min(0)

            # crop
            xyz, valid_idxs = self.crop(xyz)

            xyz_middle = xyz_middle[valid_idxs]
            xyz = xyz[valid_idxs]
            rgb = rgb[valid_idxs]
            label = label[valid_idxs]
            instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)

            # get instance information
            inst_num, inst_infos = self.getInstanceInfo(xyz_middle, instance_label.astype(np.int32), label)
            inst_info = inst_infos["instance_info"]  # (n, 9), (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
            inst_pointnum = inst_infos["instance_pointnum"]   # (nInst), list
            inst_cls = inst_infos["instance_cls"]

            instance_label[np.where(instance_label != -100)] += total_inst_num
            total_inst_num += inst_num

            # merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + xyz.shape[0])

            locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()], 1))
            locs_float.append(torch.from_numpy(xyz_middle))
            feats.append(torch.from_numpy(rgb) + torch.randn(3) * 0.1)
            labels.append(torch.from_numpy(label))
            instance_labels.append(torch.from_numpy(instance_label))

            instance_infos.append(torch.from_numpy(inst_info))
            instance_pointnum.extend(inst_pointnum)
            instance_cls.extend(inst_cls)

        # merge all the scenes in the batchd
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)

        locs = torch.cat(locs, 0)                                # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        locs_float = torch.cat(locs_float, 0).to(torch.float32)  # float (N, 3)
        feats = torch.cat(feats, 0)                              # float (N, C)
        labels = torch.cat(labels, 0).long()                     # long (N)
        instance_labels = torch.cat(instance_labels, 0).long()   # long (N)

        instance_infos = torch.cat(instance_infos, 0).to(torch.float32)       # float (N, 9) (meanxyz, minxyz, maxxyz)
        instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)  # int (total_nInst)
        instance_cls = torch.tensor(instance_cls, dtype=torch.long)            # long (total_nInst)

        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)     # long (3)

        # voxelize
        voxel_locs, p2v_map, v2p_map = softgroup_ops.voxelization_idx(locs, self.batch_size, self.mode)

        return {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
                'locs_float': locs_float, 'feats': feats, 'labels': labels, 'instance_labels': instance_labels,
                'instance_info': instance_infos, 'instance_pointnum': instance_pointnum, 'instance_cls': instance_cls,
                'id': id, 'offsets': batch_offsets, 'spatial_shape': spatial_shape}


    def valMerge(self, id):
        locs = []
        locs_float = []
        feats = []
        labels = []
        instance_labels = []

        instance_infos = []  # (N, 9)
        instance_pointnum = []  # (total_nInst), int
        instance_cls = [] # (total_nInst), long 

        batch_offsets = [0]

        total_inst_num = 0
        for i, idx in enumerate(id):
            xyz_origin, rgb, label, instance_label = self.val_files[idx]

            # flip x / rotation
            xyz_middle = self.dataAugment(xyz_origin, False, True, True)

            # scale
            xyz = xyz_middle * self.scale

            # offset
            xyz -= xyz.min(0)

            # crop
            xyz, valid_idxs = self.crop(xyz)

            xyz_middle = xyz_middle[valid_idxs]
            xyz = xyz[valid_idxs]
            rgb = rgb[valid_idxs]
            label = label[valid_idxs]
            instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)

            # get instance information
            inst_num, inst_infos = self.getInstanceInfo(xyz_middle, instance_label.astype(np.int32), label)
            inst_info = inst_infos["instance_info"]  # (n, 9), (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
            inst_pointnum = inst_infos["instance_pointnum"]  # (nInst), list
            inst_cls = inst_infos["instance_cls"]

            instance_label[np.where(instance_label != -100)] += total_inst_num
            total_inst_num += inst_num

            # merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + xyz.shape[0])

            locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()], 1))
            locs_float.append(torch.from_numpy(xyz_middle))
            feats.append(torch.from_numpy(rgb))
            labels.append(torch.from_numpy(label))
            instance_labels.append(torch.from_numpy(instance_label))

            instance_infos.append(torch.from_numpy(inst_info))
            instance_pointnum.extend(inst_pointnum)
            instance_cls.extend(inst_cls)

        # merge all the scenes in the batch
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)

        locs = torch.cat(locs, 0)                                  # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        locs_float = torch.cat(locs_float, 0).to(torch.float32)    # float (N, 3)
        feats = torch.cat(feats, 0)                                # float (N, C)
        labels = torch.cat(labels, 0).long()                       # long (N)
        instance_labels = torch.cat(instance_labels, 0).long()     # long (N)

        instance_infos = torch.cat(instance_infos, 0).to(torch.float32)               # float (N, 9) (meanxyz, minxyz, maxxyz)
        instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)          # int (total_nInst)
        instance_cls = torch.tensor(instance_cls, dtype=torch.long)            # long (total_nInst)

        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)  # long (3)

        # voxelize
        voxel_locs, p2v_map, v2p_map = softgroup_ops.voxelization_idx(locs, self.batch_size, self.mode)

        return {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
                'locs_float': locs_float, 'feats': feats, 'labels': labels, 'instance_labels': instance_labels,
                'instance_info': instance_infos, 'instance_pointnum': instance_pointnum, 'instance_cls': instance_cls,
                'id': id, 'offsets': batch_offsets, 'spatial_shape': spatial_shape}


    def testMerge(self, id):
        locs = []
        locs_float = []
        feats = []

        labels = []#

        batch_offsets = [0]
        for i, idx in enumerate(id):

            if self.test_split == 'val':
                xyz_origin, rgb, label, instance_label = self.test_files[idx]
            elif self.test_split == 'test':
                xyz_origin, rgb = self.test_files[idx]
            else:
                print("Wrong test split: {}!".format(self.test_split))
                exit(0)

            # flip x / rotation
            xyz_middle = self.dataAugment(xyz_origin, False, True, True)

            # scale
            xyz = xyz_middle * self.scale

            # offset
            xyz -= xyz.min(0)

            # merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + xyz.shape[0])

            locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()], 1))
            locs_float.append(torch.from_numpy(xyz_middle))
            feats.append(torch.from_numpy(rgb))

            if self.test_split == 'val':
                labels.append(torch.from_numpy(label))

        if self.test_split == 'val':
            labels = torch.cat(labels, 0).long()                     # long (N)

        # merge all the scenes in the batch
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)

        locs = torch.cat(locs, 0)                                         # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        locs_float = torch.cat(locs_float, 0).to(torch.float32)           # float (N, 3)
        feats = torch.cat(feats, 0)                                       # float (N, C)

        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)  # long (3)

        # voxelize
        voxel_locs, p2v_map, v2p_map = softgroup_ops.voxelization_idx(locs, self.batch_size, self.mode)

        if self.test_split == 'val':
            return {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
                    'locs_float': locs_float, 'feats': feats,
                    'id': id, 'offsets': batch_offsets, 'spatial_shape': spatial_shape,
                    'labels': labels}
        
        elif self.test_split == 'test':
            return {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
                    'locs_float': locs_float, 'feats': feats,
                    'id': id, 'offsets': batch_offsets, 'spatial_shape': spatial_shape} 
        else:
            assert Exception

