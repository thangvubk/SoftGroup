import argparse
import datetime
import os
import os.path as osp
import shutil
import time

import torch
import yaml
from munch import Munch
from softgroup.data import build_dataloader, build_dataset
from softgroup.evaluation import ScanNetEval, evaluate_semantic_acc, evaluate_semantic_miou
from softgroup.model import SoftGroup
from softgroup.util import (AverageMeter, build_optimizer, checkpoint_save, cosine_lr_after_step,
                            get_max_memory, get_root_logger, init_dist, is_multiple, is_power2,
                            load_checkpoint)
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser('SoftGroup')
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('--dist', action='store_true', help='run with distributed parallel')
    parser.add_argument('--resume', type=str, help='path to resume from')
    parser.add_argument('--work_dir', type=str, help='working directory')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    cfg_txt = open(args.config, 'r').read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))

    if args.dist:
        init_dist()

    # work_dir & logger
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    os.makedirs(osp.abspath(cfg.work_dir), exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file)
    logger.info(f'Config:\n{cfg_txt}')
    logger.info(f'Distributed: {args.dist}')
    shutil.copy(args.config, osp.join(cfg.work_dir, osp.basename(args.config)))
    writer = SummaryWriter(cfg.work_dir)

    # model
    model = SoftGroup(**cfg.model).cuda()
    if args.dist:
        model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])

    # data
    train_set = build_dataset(cfg.data.train, logger)
    val_set = build_dataset(cfg.data.test, logger)
    train_loader = build_dataloader(
        train_set, training=True, dist=args.dist, **cfg.dataloader.train)
    val_loader = build_dataloader(val_set, training=False, **cfg.dataloader.test)

    # optim
    optimizer = build_optimizer(model, cfg.optimizer)

    # pretrain, resume
    start_epoch = 1
    if args.resume:
        logger.info(f'Resume from {args.resume}')
        start_epoch = load_checkpoint(args.resume, logger, model, optimizer=optimizer)
    elif cfg.pretrain:
        logger.info(f'Load pretrain from {cfg.pretrain}')
        load_checkpoint(cfg.pretrain, logger, model)

    # train and val
    logger.info('Training')
    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        iter_time = AverageMeter()
        data_time = AverageMeter()
        meter_dict = {}
        end = time.time()

        if train_loader.sampler is not None and args.dist:
            train_loader.sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader, start=1):
            data_time.update(time.time() - end)

            cosine_lr_after_step(optimizer, cfg.optimizer.lr, epoch - 1, cfg.step_epoch, cfg.epochs)
            loss, log_vars = model(batch, return_loss=True)

            # meter_dict
            for k, v in log_vars.items():
                if k not in meter_dict.keys():
                    meter_dict[k] = AverageMeter()
                meter_dict[k].update(v[0], v[1])

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # time and print
            current_iter = (epoch - 1) * len(train_loader) + i
            max_iter = cfg.epochs * len(train_loader)
            remain_iter = max_iter - current_iter
            iter_time.update(time.time() - end)
            end = time.time()
            remain_time = remain_iter * iter_time.avg
            remain_time = str(datetime.timedelta(seconds=int(remain_time)))
            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('learning_rate', lr, current_iter)
            for k, v in meter_dict.items():
                writer.add_scalar(k, v.val, current_iter)
            if is_multiple(i, 10):
                log_str = f'Epoch [{epoch}/{cfg.epochs}][{i}/{len(train_loader)}]  '
                log_str += f'lr: {lr:.2g}, eta: {remain_time}, mem: {get_max_memory()}, '\
                    f'data_time: {data_time.val:.2f}, iter_time: {iter_time.val:.2f}'
                for k, v in meter_dict.items():
                    log_str += f', {k}: {v.val:.4f}'
                logger.info(log_str)
        checkpoint_save(epoch, model, optimizer, cfg.work_dir, cfg.save_freq)

        # validation
        if is_multiple(epoch, cfg.save_freq) or is_power2(epoch):
            all_sem_preds, all_sem_labels, all_pred_insts, all_gt_insts = [], [], [], []
            logger.info('Validation')
            with torch.no_grad():
                model = model.eval()
                for batch in tqdm(val_loader, total=len(val_loader)):
                    ret = model(batch)
                    all_sem_preds.append(ret['semantic_preds'])
                    all_sem_labels.append(ret['semantic_labels'])
                    if not cfg.model.semantic_only:
                        all_pred_insts.append(ret['pred_instances'])
                        all_gt_insts.append(ret['gt_instances'])
                if not cfg.model.semantic_only:
                    logger.info('Evaluate instance segmentation')
                    scannet_eval = ScanNetEval(val_loader.dataset.CLASSES)
                    scannet_eval.evaluate(all_pred_insts, all_gt_insts)
                logger.info('Evaluate semantic segmentation')
                miou = evaluate_semantic_miou(all_sem_preds, all_sem_labels, cfg.model.ignore_label,
                                              logger)
                acc = evaluate_semantic_acc(all_sem_preds, all_sem_labels, cfg.model.ignore_label,
                                            logger)
                writer.add_scalar('mIoU', miou, epoch)
                writer.add_scalar('Acc', acc, epoch)
        writer.flush()
