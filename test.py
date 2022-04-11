import argparse
import random

import numpy as np
import torch
import yaml
from munch import Munch
from softgroup.data import build_dataloader, build_dataset
from softgroup.evaluation import ScanNetEval, evaluate_semantic_acc, evaluate_semantic_miou
from softgroup.model import SoftGroup
from softgroup.util import get_root_logger, load_checkpoint
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser('SoftGroup')
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('checkpoint', type=str, help='path to checkpoint')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    torch.backends.cudnn.enabled = False  # TODO remove this
    test_seed = 567
    random.seed(test_seed)
    np.random.seed(test_seed)
    torch.manual_seed(test_seed)
    torch.cuda.manual_seed_all(test_seed)

    args = get_args()
    cfg_txt = open(args.config, 'r').read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
    logger = get_root_logger()

    model = SoftGroup(**cfg.model).cuda()
    logger.info(f'Load state dict from {args.checkpoint}')
    load_checkpoint(args.checkpoint, logger, model)

    dataset = build_dataset(cfg.data.test, logger)
    dataloader = build_dataloader(dataset, training=False, **cfg.dataloader.test)
    all_sem_preds, all_sem_labels, all_pred_insts, all_gt_insts = [], [], [], []
    with torch.no_grad():
        model = model.eval()
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            ret = model(batch)
            all_sem_preds.append(ret['semantic_preds'])
            all_sem_labels.append(ret['semantic_labels'])
            if not cfg.model.semantic_only:
                all_pred_insts.append(ret['pred_instances'])
                all_gt_insts.append(ret['gt_instances'])
        if not cfg.model.semantic_only:
            logger.info('Evaluate instance segmentation')
            scannet_eval = ScanNetEval(dataset.CLASSES)
            scannet_eval.evaluate(all_pred_insts, all_gt_insts)
        logger.info('Evaluate semantic segmentation')
        evaluate_semantic_miou(all_sem_preds, all_sem_labels, cfg.model.ignore_label, logger)
        evaluate_semantic_acc(all_sem_preds, all_sem_labels, cfg.model.ignore_label, logger)
