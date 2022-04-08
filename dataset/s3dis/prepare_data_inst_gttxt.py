"""Generate instance groundtruth .txt files (for evaluation) modified by thang:

fix label id.
"""

import argparse
import glob
import os

import numpy as np
import torch

# import gorilla

INV_OBJECT_LABEL = {
    0: 'ceiling',
    1: 'floor',
    2: 'wall',
    3: 'beam',
    4: 'column',
    5: 'window',
    6: 'door',
    7: 'chair',
    8: 'table',
    9: 'bookcase',
    10: 'sofa',
    11: 'board',
    12: 'clutter',
}
semantic_label_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]


def get_parser():
    parser = argparse.ArgumentParser(description='s3dis data prepare')
    parser.add_argument(
        '--data-dir', type=str, default='./preprocess', help='directory save processed data')
    parser.add_argument(
        '--save-dir', type=str, default='./val_gt', help='directory save ground truth')

    args_cfg = parser.parse_args()

    return args_cfg


if __name__ == '__main__':
    args = get_parser()
    data_dir = args.data_dir
    save_dir = args.save_dir
    files = sorted(glob.glob(f'{data_dir}/*.pth'))
    # rooms = [torch.load(i) for i in gorilla.track(files)]

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for i, f in enumerate(files):
        (xyz, rgb, semantic_labels, instance_labels, room_label,
         scene) = torch.load(f)  # semantic label 0-12 instance_labels 0~instance_num-1 -100
        print(f'{i + 1}/{len(files)} {scene}')

        instance_labels_new = np.zeros(
            instance_labels.shape, dtype=np.int32
        )  # 0 for unannotated, xx00y: x for semantic_label, y for inst_id (1~instance_num)

        instance_num = int(instance_labels.max()) + 1
        inst_ids = np.unique(instance_labels)
        for inst_id in inst_ids:
            if inst_id <= 0:
                continue
            instance_mask = np.where(instance_labels == inst_id)[0]
            sem_id = int(semantic_labels[instance_mask[0]])
            if (sem_id == -100):
                sem_id = 0
            semantic_label = semantic_label_idxs[sem_id]
            instance_labels_new[instance_mask] = semantic_label * 1000 + inst_id

        np.savetxt(os.path.join(save_dir, scene + '.txt'), instance_labels_new, fmt='%d')
