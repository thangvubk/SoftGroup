# Copyright (c) Gorilla-Lab. All rights reserved.
import argparse
import glob
import os
import os.path as osp
from random import sample

import numpy as np
import torch


def random_sample(coords: np.ndarray, colors: np.ndarray, semantic_labels: np.ndarray,
                  instance_labels: np.ndarray, ratio: float):
    num_points = coords.shape[0]
    num_sample = int(num_points * ratio)
    sample_ids = sample(range(num_points), num_sample)

    # downsample
    coords = coords[sample_ids]
    colors = colors[sample_ids]
    semantic_labels = semantic_labels[sample_ids]
    instance_labels = instance_labels[sample_ids]

    return coords, colors, semantic_labels, instance_labels


def get_parser():
    parser = argparse.ArgumentParser(description='downsample s3dis by voxelization')
    parser.add_argument(
        '--data-dir', type=str, default='./preprocess', help='directory save processed data')
    parser.add_argument('--ratio', type=float, default=0.25, help='random downsample ratio')
    parser.add_argument(
        '--voxel-size',
        type=float,
        default=None,
        help='voxelization size (priority is higher than voxel-size)')
    parser.add_argument('--verbose', action='store_true', help='show partition information or not')

    args_cfg = parser.parse_args()

    return args_cfg


if __name__ == '__main__':
    args = get_parser()

    data_dir = args.data_dir
    # voxelize or not
    voxelize_flag = args.voxel_size is not None
    if voxelize_flag:
        print('processing: voxelize')
        save_dir = f'{data_dir}_voxelize'
    else:
        print('processing: random sample')
        save_dir = f'{data_dir}_sample'
    os.makedirs(save_dir, exist_ok=True)

    # for data_file in [osp.join(data_dir, "Area_6_office_17.pth")]:
    for data_file in glob.glob(osp.join(data_dir, '*.pth')):
        # for data_file in glob.glob(osp.join(data_dir, "*.pth")):
        (coords, colors, semantic_labels, instance_labels, room_label,
         scene) = torch.load(data_file)

        if args.verbose:
            print(f'processing: {scene}')

        save_path = osp.join(save_dir, f'{scene}_inst_nostuff.pth')
        if os.path.exists(save_path):
            continue

        if voxelize_flag:
            raise NotImplementedError
        else:
            coords, colors, semantic_labels, instance_labels = \
                random_sample(coords, colors, semantic_labels, instance_labels, args.ratio)

        torch.save((coords, colors, semantic_labels, instance_labels, room_label, scene), save_path)
