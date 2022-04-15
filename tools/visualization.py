import argparse
import os
from operator import itemgetter

import numpy as np
import torch

# yapf:disable
COLOR_DETECTRON2 = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        # 0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        # 0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        # 0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        # 0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        # 0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.857, 0.857, 0.857,
        # 1.000, 1.000, 1.000
    ]).astype(np.float32).reshape(-1, 3) * 255
# yapf:enable

SEMANTIC_IDXS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
SEMANTIC_NAMES = np.array([
    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
    'picture', 'counter', 'desk', 'curtain', 'refridgerator', 'shower curtain', 'toilet', 'sink',
    'bathtub', 'otherfurniture'
])
CLASS_COLOR = {
    'unannotated': [0, 0, 0],
    'floor': [143, 223, 142],
    'wall': [171, 198, 230],
    'cabinet': [0, 120, 177],
    'bed': [255, 188, 126],
    'chair': [189, 189, 57],
    'sofa': [144, 86, 76],
    'table': [255, 152, 153],
    'door': [222, 40, 47],
    'window': [197, 176, 212],
    'bookshelf': [150, 103, 185],
    'picture': [200, 156, 149],
    'counter': [0, 190, 206],
    'desk': [252, 183, 210],
    'curtain': [219, 219, 146],
    'refridgerator': [255, 127, 43],
    'bathtub': [234, 119, 192],
    'shower curtain': [150, 218, 228],
    'toilet': [0, 160, 55],
    'sink': [110, 128, 143],
    'otherfurniture': [80, 83, 160]
}
SEMANTIC_IDX2NAME = {
    1: 'wall',
    2: 'floor',
    3: 'cabinet',
    4: 'bed',
    5: 'chair',
    6: 'sofa',
    7: 'table',
    8: 'door',
    9: 'window',
    10: 'bookshelf',
    11: 'picture',
    12: 'counter',
    14: 'desk',
    16: 'curtain',
    24: 'refridgerator',
    28: 'shower curtain',
    33: 'toilet',
    34: 'sink',
    36: 'bathtub',
    39: 'otherfurniture'
}


def get_coords_color(opt):
    if opt.dataset == 's3dis':
        assert opt.data_split in ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_5', 'Area_6'],\
            'data_split for s3dis should be one of [Area_1, Area_2, Area_3, Area_4, Area_5, Area_6]'
        input_file = os.path.join('dataset', opt.dataset, 'preprocess',
                                  opt.room_name + '_inst_nostuff.pth')
        assert os.path.isfile(input_file), 'File not exist - {}.'.format(input_file)
        xyz, rgb, label, inst_label, _, _ = torch.load(input_file)
        # update variable to match scannet format
        opt.data_split = os.path.join('val', opt.data_split)
    else:
        input_file = os.path.join('dataset', opt.dataset, opt.data_split,
                                  opt.room_name + '_inst_nostuff.pth')
        assert os.path.isfile(input_file), 'File not exist - {}.'.format(input_file)
        if opt.data_split == 'test':
            xyz, rgb = torch.load(input_file)
        else:
            xyz, rgb, label, inst_label = torch.load(input_file)

    rgb = (rgb + 1) * 127.5

    if (opt.task == 'semantic_gt'):
        assert opt.data_split != 'test'
        label = label.astype(np.int)
        label_rgb = np.zeros(rgb.shape)
        label_rgb[label >= 0] = np.array(
            itemgetter(*SEMANTIC_NAMES[label[label >= 0]])(CLASS_COLOR))
        rgb = label_rgb

    elif (opt.task == 'semantic_pred'):
        assert opt.data_split != 'train'
        semantic_file = os.path.join(opt.prediction_path, opt.data_split, 'semantic',
                                     opt.room_name + '.npy')
        assert os.path.isfile(semantic_file), 'No semantic result - {}.'.format(semantic_file)
        label_pred = np.load(semantic_file).astype(np.int)  # 0~19
        label_pred_rgb = np.array(itemgetter(*SEMANTIC_NAMES[label_pred])(CLASS_COLOR))
        rgb = label_pred_rgb

    elif (opt.task == 'offset_semantic_pred'):
        assert opt.data_split != 'train'
        semantic_file = os.path.join(opt.prediction_path, opt.data_split, 'semantic',
                                     opt.room_name + '.npy')
        assert os.path.isfile(semantic_file), 'No semantic result - {}.'.format(semantic_file)
        label_pred = np.load(semantic_file).astype(np.int)  # 0~19
        label_pred_rgb = np.array(itemgetter(*SEMANTIC_NAMES[label_pred])(CLASS_COLOR))
        rgb = label_pred_rgb

        offset_file = os.path.join(opt.prediction_path, opt.data_split, 'coords_offsets',
                                   opt.room_name + '.npy')
        assert os.path.isfile(offset_file), 'No offset result - {}.'.format(offset_file)
        offset_coords = np.load(offset_file)
        xyz = offset_coords[:, :3] + offset_coords[:, 3:]

    # same color order according to instance pointnum
    elif (opt.task == 'instance_gt'):
        assert opt.data_split != 'test'
        inst_label = inst_label.astype(np.int)
        print('Instance number: {}'.format(inst_label.max() + 1))
        inst_label_rgb = np.zeros(rgb.shape)
        ins_num = inst_label.max() + 1
        ins_pointnum = np.zeros(ins_num)
        for _ins_id in range(ins_num):
            ins_pointnum[_ins_id] = (inst_label == _ins_id).sum()
        sort_idx = np.argsort(ins_pointnum)[::-1]
        for _sort_id in range(ins_num):
            inst_label_rgb[inst_label == sort_idx[_sort_id]] = COLOR_DETECTRON2[
                _sort_id % len(COLOR_DETECTRON2)]
        rgb = inst_label_rgb

    # same color order according to instance pointnum
    elif (opt.task == 'instance_pred'):
        assert opt.data_split != 'train'
        instance_file = os.path.join(opt.prediction_path, opt.data_split, opt.room_name + '.txt')
        assert os.path.isfile(instance_file), 'No instance result - {}.'.format(instance_file)
        f = open(instance_file, 'r')
        masks = f.readlines()
        masks = [mask.rstrip().split() for mask in masks]
        inst_label_pred_rgb = np.zeros(rgb.shape)  # np.ones(rgb.shape) * 255 #

        ins_num = len(masks)
        ins_pointnum = np.zeros(ins_num)
        inst_label = -100 * np.ones(rgb.shape[0]).astype(np.int)

        # sort score such that high score has high priority for visualization
        scores = np.array([float(x[-1]) for x in masks])
        sort_inds = np.argsort(scores)[::-1]
        for i_ in range(len(masks) - 1, -1, -1):
            i = sort_inds[i_]
            mask_path = os.path.join(opt.prediction_path, opt.data_split, masks[i][0])
            assert os.path.isfile(mask_path), mask_path
            if (float(masks[i][2]) < 0.09):
                continue
            mask = np.loadtxt(mask_path).astype(np.int)
            if opt.dataset == 'scannet':
                print('{} {}: {} pointnum: {}'.format(i,
                                                      masks[i], SEMANTIC_IDX2NAME[int(masks[i][1])],
                                                      mask.sum()))
            else:
                print('{} {}: pointnum: {}'.format(i, masks[i], mask.sum()))
            ins_pointnum[i] = mask.sum()
            inst_label[mask == 1] = i
        sort_idx = np.argsort(ins_pointnum)[::-1]
        for _sort_id in range(ins_num):
            inst_label_pred_rgb[inst_label == sort_idx[_sort_id]] = COLOR_DETECTRON2[
                _sort_id % len(COLOR_DETECTRON2)]
        rgb = inst_label_pred_rgb

    if opt.data_split != 'test':
        sem_valid = (label != -100)
        xyz = xyz[sem_valid]
        rgb = rgb[sem_valid]

    return xyz, rgb


def write_ply(verts, colors, indices, output_file):
    if colors is None:
        colors = np.zeros_like(verts)
    if indices is None:
        indices = []

    file = open(output_file, 'w')
    file.write('ply \n')
    file.write('format ascii 1.0\n')
    file.write('element vertex {:d}\n'.format(len(verts)))
    file.write('property float x\n')
    file.write('property float y\n')
    file.write('property float z\n')
    file.write('property uchar red\n')
    file.write('property uchar green\n')
    file.write('property uchar blue\n')
    file.write('element face {:d}\n'.format(len(indices)))
    file.write('property list uchar uint vertex_indices\n')
    file.write('end_header\n')
    for vert, color in zip(verts, colors):
        file.write('{:f} {:f} {:f} {:d} {:d} {:d}\n'.format(vert[0], vert[1], vert[2],
                                                            int(color[0] * 255),
                                                            int(color[1] * 255),
                                                            int(color[2] * 255)))
    for ind in indices:
        file.write('3 {:d} {:d} {:d}\n'.format(ind[0], ind[1], ind[2]))
    file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        choices=['scannet', 's3dis'],
        help='dataset for visualization',
        default='scannet')
    parser.add_argument(
        '--prediction_path',
        help='path to the prediction results',
        default='./exp/scannetv2/softgroup/softgroup_default_scannet/result')
    parser.add_argument(
        '--data_split', help='train/val/test for scannet or Area_ID for s3dis', default='val')
    parser.add_argument('--room_name', help='room_name', default='scene0011_00')
    parser.add_argument(
        '--task',
        help='input/semantic_gt/semantic_pred/offset_semantic_pred/instance_gt/instance_pred',
        default='instance_pred')
    parser.add_argument('--out', help='output point cloud file in FILE.ply format')
    opt = parser.parse_args()

    xyz, rgb = get_coords_color(opt)
    points = xyz[:, :3]
    colors = rgb / 255

    if opt.out != '':
        assert '.ply' in opt.out, 'output cloud file should be in FILE.ply format'
        write_ply(points, colors, None, opt.out)
    else:
        import open3d as o3d
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)
        pc.colors = o3d.utility.Vector3dVector(colors)

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pc)
        vis.get_render_option().point_size = 1.5
        vis.run()
        vis.destroy_window()
