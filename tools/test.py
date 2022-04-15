import argparse

import torch
import yaml
from munch import Munch
from softgroup.data import build_dataloader, build_dataset
from softgroup.evaluation import (ScanNetEval, evaluate_offset_mae, evaluate_semantic_acc,
                                  evaluate_semantic_miou)
from softgroup.model import SoftGroup
from softgroup.util import (collect_results_gpu, get_dist_info, get_root_logger, init_dist,
                            is_main_process, load_checkpoint)
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser('SoftGroup')
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('checkpoint', type=str, help='path to checkpoint')
    parser.add_argument('--dist', action='store_true', help='run with distributed parallel')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
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
    all_sem_preds, all_sem_labels, all_offset_preds, all_offset_labels = [], [], [], []
    all_inst_labels, all_pred_insts, all_gt_insts = [], [], []
    _, world_size = get_dist_info()
    progress_bar = tqdm(total=len(dataloader) * world_size, disable=not is_main_process())
    with torch.no_grad():
        model = model.eval()
        for i, batch in enumerate(dataloader):
            result = model(batch)
            results.append(result)
            progress_bar.update(world_size)
        progress_bar.close()
        results = collect_results_gpu(results, len(dataset))
    if is_main_process():
        for res in results:
            all_sem_preds.append(res['semantic_preds'])
            all_sem_labels.append(res['semantic_labels'])
            all_offset_preds.append(res['offset_preds'])
            all_offset_labels.append(res['offset_labels'])
            all_inst_labels.append(res['instance_labels'])
            if not cfg.model.semantic_only:
                all_pred_insts.append(res['pred_instances'])
                all_gt_insts.append(res['gt_instances'])
        if not cfg.model.semantic_only:
            logger.info('Evaluate instance segmentation')
            scannet_eval = ScanNetEval(dataset.CLASSES)
            scannet_eval.evaluate(all_pred_insts, all_gt_insts)
        logger.info('Evaluate semantic segmentation and offset MAE')
        evaluate_semantic_miou(all_sem_preds, all_sem_labels, cfg.model.ignore_label, logger)
        evaluate_semantic_acc(all_sem_preds, all_sem_labels, cfg.model.ignore_label, logger)
        evaluate_offset_mae(all_offset_preds, all_offset_labels, all_inst_labels,
                            cfg.model.ignore_label, logger)
