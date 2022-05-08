import argparse
import multiprocessing as mp
import os
import os.path as osp
from functools import partial

import numpy as np
import torch
import yaml
from munch import Munch
from softgroup.data import build_dataloader, build_dataset
from softgroup.evaluation import (ScanNetEval, evaluate_offset_mae, evaluate_semantic_acc,
                                  evaluate_semantic_miou)
from softgroup.model import SoftGroup
from softgroup.util import (collect_results_gpu, get_dist_info, get_root_logger, init_dist,
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


def save_single_instance(root, scan_id, insts):
    f = open(osp.join(root, f'{scan_id}.txt'), 'w')
    os.makedirs(osp.join(root, 'predicted_masks'), exist_ok=True)
    for i, inst in enumerate(insts):
        assert scan_id == inst['scan_id']
        label_id = inst['label_id']
        conf = inst['conf']
        f.write(f'predicted_masks/{scan_id}_{i:03d}.txt {label_id} {conf:.4f}\n')
        mask_path = osp.join(root, 'predicted_masks', f'{scan_id}_{i:03d}.txt')
        mask = rle_decode(inst['pred_mask'])
        np.savetxt(mask_path, mask, fmt='%d')
    f.close()


def save_pred_instances(root, name, scan_ids, pred_insts):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    roots = [root] * len(scan_ids)
    pool = mp.Pool()
    pool.starmap(save_single_instance, zip(roots, scan_ids, pred_insts))
    pool.close()
    pool.join()


def save_gt_instances(root, name, scan_ids, gt_insts):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    paths = [osp.join(root, f'{i}.txt') for i in scan_ids]
    pool = mp.Pool()
    map_func = partial(np.savetxt, fmt='%d')
    pool.starmap(map_func, zip(paths, gt_insts))
    pool.close()
    pool.join()


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
    scan_ids, coords, sem_preds, sem_labels, offset_preds, offset_labels = [], [], [], [], [], []
    inst_labels, pred_insts, gt_insts = [], [], []
    _, world_size = get_dist_info()
    progress_bar = tqdm(total=len(dataloader) * world_size, disable=not is_main_process())
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(dataloader):
            result = model(batch)
            results.append(result)
            progress_bar.update(world_size)
        progress_bar.close()
        results = collect_results_gpu(results, len(dataset))
    if is_main_process():
        for res in results:
            scan_ids.append(res['scan_id'])
            coords.append(res['coords_float'])
            sem_preds.append(res['semantic_preds'])
            sem_labels.append(res['semantic_labels'])
            offset_preds.append(res['offset_preds'])
            offset_labels.append(res['offset_labels'])
            inst_labels.append(res['instance_labels'])
            if not cfg.model.semantic_only:
                pred_insts.append(res['pred_instances'])
                gt_insts.append(res['gt_instances'])
        if not cfg.model.semantic_only:
            logger.info('Evaluate instance segmentation')
            scannet_eval = ScanNetEval(dataset.CLASSES)
            scannet_eval.evaluate(pred_insts, gt_insts)
        logger.info('Evaluate semantic segmentation and offset MAE')
        ignore_label = cfg.model.ignore_label
        evaluate_semantic_miou(sem_preds, sem_labels, ignore_label, logger)
        evaluate_semantic_acc(sem_preds, sem_labels, ignore_label, logger)
        evaluate_offset_mae(offset_preds, offset_labels, inst_labels, ignore_label, logger)

        # save output
        if not args.out:
            return
        logger.info('Save results')
        save_npy(args.out, 'coords', scan_ids, coords)
        if cfg.save_cfg.semantic:
            save_npy(args.out, 'semantic_pred', scan_ids, sem_preds)
            save_npy(args.out, 'semantic_label', scan_ids, sem_labels)
        if cfg.save_cfg.offset:
            save_npy(args.out, 'offset_pred', scan_ids, offset_preds)
            save_npy(args.out, 'offset_label', scan_ids, offset_labels)
        if cfg.save_cfg.instance:
            save_pred_instances(args.out, 'pred_instance', scan_ids, pred_insts)
            save_gt_instances(args.out, 'gt_instance', scan_ids, gt_insts)


if __name__ == '__main__':
    main()
