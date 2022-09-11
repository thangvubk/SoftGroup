import argparse
import multiprocessing as mp
import os
import os.path as osp

import numpy as np
import torch
import yaml
from munch import Munch
from softgroup.data import build_dataloader, build_dataset
from softgroup.evaluation import (PanopticEval, ScanNetEval, evaluate_offset_mae,
                                  evaluate_semantic_acc, evaluate_semantic_miou)
from softgroup.model import SoftGroup
from softgroup.util import (collect_results_cpu, get_dist_info, get_root_logger, init_dist,
                            is_main_process, load_checkpoint, rle_decode)
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser('SoftGroup')
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('checkpoint', type=str, help='path to checkpoint')
    parser.add_argument('--dist', action='store_true', help='run with distributed parallel')
    parser.add_argument('--out', type=str, help='directory for output results')
    args = parser.parse_args()
    return args


def save_npy(root, name, scan_ids, arrs):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    paths = [osp.join(root, f'{i}.npy') for i in scan_ids]
    pool = mp.Pool()
    pool.starmap(np.save, zip(paths, arrs))
    pool.close()
    pool.join()


def save_single_instance(root, scan_id, insts, nyu_id=None):
    f = open(osp.join(root, f'{scan_id}.txt'), 'w')
    os.makedirs(osp.join(root, 'predicted_masks'), exist_ok=True)
    for i, inst in enumerate(insts):
        assert scan_id == inst['scan_id']
        label_id = inst['label_id']
        # scannet dataset use nyu_id for evaluation
        if nyu_id is not None:
            label_id = nyu_id[label_id - 1]
        conf = inst['conf']
        f.write(f'predicted_masks/{scan_id}_{i:03d}.txt {label_id} {conf:.4f}\n')
        mask_path = osp.join(root, 'predicted_masks', f'{scan_id}_{i:03d}.txt')
        mask = rle_decode(inst['pred_mask'])
        np.savetxt(mask_path, mask, fmt='%d')
    f.close()


def save_pred_instances(root, name, scan_ids, pred_insts, nyu_id=None):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    roots = [root] * len(scan_ids)
    nyu_ids = [nyu_id] * len(scan_ids)
    pool = mp.Pool()
    pool.starmap(save_single_instance, zip(roots, scan_ids, pred_insts, nyu_ids))
    pool.close()
    pool.join()


def save_gt_instance(path, gt_inst, nyu_id=None):
    if nyu_id is not None:
        sem = gt_inst // 1000
        ignore = sem == 0
        ins = gt_inst % 1000
        nyu_id = np.array(nyu_id)
        sem = nyu_id[sem - 1]
        sem[ignore] = 0
        gt_inst = sem * 1000 + ins
    np.savetxt(path, gt_inst, fmt='%d')


def save_gt_instances(root, name, scan_ids, gt_insts, nyu_id=None):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    paths = [osp.join(root, f'{i}.txt') for i in scan_ids]
    pool = mp.Pool()
    nyu_ids = [nyu_id] * len(scan_ids)
    pool.starmap(save_gt_instance, zip(paths, gt_insts, nyu_ids))
    pool.close()
    pool.join()


def save_panoptic_single(path, panoptic_pred, learning_map_inv, num_classes):
    # convert cls to kitti format
    panoptic_ids = panoptic_pred >> 16
    panoptic_cls = panoptic_pred & 0xFFFF
    new_learning_map_inv = {num_classes: 0}
    for k, v in learning_map_inv.items():
        if k == 0:
            continue
        if k < 9:
            new_k = k + 10
        else:
            new_k = k - 9
        new_learning_map_inv[new_k] = v
    panoptic_cls = np.vectorize(new_learning_map_inv.__getitem__)(panoptic_cls).astype(
        panoptic_pred.dtype)
    panoptic_pred = (panoptic_cls & 0xFFFF) | (panoptic_ids << 16)
    panoptic_pred.tofile(path)


def save_panoptic(root, name, scan_ids, arrs, learning_map_inv, num_classes):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    paths = [osp.join(root, f'{i}.label'.replace('velodyne', 'predictions')) for i in scan_ids]
    learning_map_invs = [learning_map_inv] * len(scan_ids)
    num_classes_list = [num_classes] * len(scan_ids)
    for p in paths:
        os.makedirs(osp.dirname(p), exist_ok=True)
    pool = mp.Pool()
    pool.starmap(save_panoptic_single, zip(paths, arrs, learning_map_invs, num_classes_list))


def main():
    args = get_args()
    cfg_txt = open(args.config, 'r').read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
    if args.dist:
        init_dist()
    logger = get_root_logger()

    model = SoftGroup(**cfg.model).cuda()
    if args.dist:
        model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
    logger.info(f'Load state dict from {args.checkpoint}')
    load_checkpoint(args.checkpoint, logger, model)

    dataset = build_dataset(cfg.data.test, logger)
    dataloader = build_dataloader(dataset, training=False, dist=args.dist, **cfg.dataloader.test)
    results = []
    scan_ids, coords, colors, sem_preds, sem_labels = [], [], [], [], []
    offset_preds, offset_labels, inst_labels, pred_insts, gt_insts = [], [], [], [], []
    panoptic_preds = []
    _, world_size = get_dist_info()
    progress_bar = tqdm(total=len(dataloader) * world_size, disable=not is_main_process())
    eval_tasks = cfg.model.test_cfg.eval_tasks
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(dataloader):
            result = model(batch)
            results.append(result)
            progress_bar.update(world_size)
        progress_bar.close()
        results = collect_results_cpu(results, len(dataset))
    if is_main_process():
        for res in results:
            scan_ids.append(res['scan_id'])
            if 'semantic' in eval_tasks or 'panoptic' in eval_tasks:
                sem_labels.append(res['semantic_labels'])
                inst_labels.append(res['instance_labels'])
            if 'semantic' in eval_tasks:
                coords.append(res['coords_float'])
                colors.append(res['color_feats'])
                sem_preds.append(res['semantic_preds'])
                offset_preds.append(res['offset_preds'])
                offset_labels.append(res['offset_labels'])
            if 'instance' in eval_tasks:
                pred_insts.append(res['pred_instances'])
                gt_insts.append(res['gt_instances'])
            if 'panoptic' in eval_tasks:
                panoptic_preds.append(res['panoptic_preds'])
        if 'instance' in eval_tasks:
            logger.info('Evaluate instance segmentation')
            eval_min_npoint = getattr(cfg, 'eval_min_npoint', None)
            scannet_eval = ScanNetEval(dataset.CLASSES, eval_min_npoint)
            scannet_eval.evaluate(pred_insts, gt_insts)
        if 'panoptic' in eval_tasks:
            logger.info('Evaluate panoptic segmentation')
            eval_min_npoint = getattr(cfg, 'eval_min_npoint', None)
            panoptic_eval = PanopticEval(dataset.THING, dataset.STUFF, min_points=eval_min_npoint)
            panoptic_eval.evaluate(panoptic_preds, sem_labels, inst_labels)
        if 'semantic' in eval_tasks:
            logger.info('Evaluate semantic segmentation and offset MAE')
            ignore_label = cfg.model.ignore_label
            evaluate_semantic_miou(sem_preds, sem_labels, ignore_label, logger)
            evaluate_semantic_acc(sem_preds, sem_labels, ignore_label, logger)
            evaluate_offset_mae(offset_preds, offset_labels, inst_labels, ignore_label, logger)

        # save output
        if not args.out:
            return
        logger.info('Save results')
        if 'semantic' in eval_tasks:
            save_npy(args.out, 'coords', scan_ids, coords)
            save_npy(args.out, 'colors', scan_ids, colors)
            save_npy(args.out, 'semantic_pred', scan_ids, sem_preds)
            save_npy(args.out, 'semantic_label', scan_ids, sem_labels)
            save_npy(args.out, 'offset_pred', scan_ids, offset_preds)
            save_npy(args.out, 'offset_label', scan_ids, offset_labels)
        if 'instance' in eval_tasks:
            nyu_id = dataset.NYU_ID
            save_pred_instances(args.out, 'pred_instance', scan_ids, pred_insts, nyu_id)
            save_gt_instances(args.out, 'gt_instance', scan_ids, gt_insts, nyu_id)
        if 'panoptic' in eval_tasks:
            save_panoptic(args.out, 'panoptic', scan_ids, panoptic_preds, dataset.learning_map_inv,
                          cfg.model.semantic_classes)


if __name__ == '__main__':
    main()
