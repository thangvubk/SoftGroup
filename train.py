import argparse
import datetime
import numpy as np
import os
import os.path as osp
import random
import shutil
import sys
import time
import torch
import yaml
from munch import Munch
from tensorboardX import SummaryWriter

from data import build_dataloader, build_dataset
from model.softgroup import SoftGroup
from util import build_optimizer, get_max_memory, get_root_logger, utils


def eval_epoch(val_loader, model, model_fn, epoch):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    am_dict = {}

    with torch.no_grad():
        model.eval()
        start_epoch = time.time()
        for i, batch in enumerate(val_loader):

            # prepare input and forward
            loss, preds, visual_dict, meter_dict = model_fn(
                batch, model, epoch, semantic_only=cfg.semantic_only)

            for k, v in meter_dict.items():
                if k not in am_dict.keys():
                    am_dict[k] = utils.AverageMeter()
                am_dict[k].update(v[0], v[1])
            sys.stdout.write("\riter: {}/{} loss: {:.4f}({:.4f})".format(
                i + 1, len(val_loader), am_dict['loss'].val, am_dict['loss'].avg))

        logger.info("epoch: {}/{}, val loss: {:.4f}, time: {}s".format(
            epoch, cfg.epochs, am_dict['loss'].avg,
            time.time() - start_epoch))

        for k in am_dict.keys():
            if k in visual_dict.keys():
                writer.add_scalar(k + '_eval', am_dict[k].avg, epoch)


def get_args():
    parser = argparse.ArgumentParser('SoftGroup')
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('--resume', type=str, help='path to resume from')
    parser.add_argument('--work_dir', type=str, help='working directory')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # TODO remove these setup
    torch.backends.cudnn.enabled = False
    test_seed = 123
    random.seed(test_seed)
    np.random.seed(test_seed)
    torch.manual_seed(test_seed)
    torch.cuda.manual_seed_all(test_seed)

    args = get_args()
    cfg_txt = open(args.config, 'r').read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))

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
    shutil.copy(args.config, osp.join(cfg.work_dir, osp.basename(args.config)))
    writer = SummaryWriter(cfg.work_dir)

    # model
    model = SoftGroup(**cfg.model).cuda()

    # data
    train_set = build_dataset(cfg.data.train, logger)
    val_set = build_dataset(cfg.data.test, logger)
    train_loader = build_dataloader(train_set, training=True, **cfg.data.dataloader)
    val_loader = build_dataloader(val_set, training=False)

    # optim
    optimizer = build_optimizer(model, cfg.optimizer)

    # pretrain, resume
    start_epoch = 1
    if args.resume:
        logger.info(f'Resume from {args.resume}')
        start_epoch = utils.load_checkpoint(args.resume, logger, model, optimizer=optimizer)
    elif cfg.pretrain:
        logger.info(f'Load pretrain from {cfg.pretrain}')
        utils.load_checkpoint(cfg.pretrain, logger, model)

    # train and val
    logger.info('Training')
    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        iter_time = utils.AverageMeter()
        data_time = utils.AverageMeter()
        meter_dict = {}
        end = time.time()

        for i, batch in enumerate(train_loader, start=1):
            data_time.update(time.time() - end)

            utils.cosine_lr_after_step(optimizer, cfg.optimizer.lr, epoch - 1, cfg.step_epoch,
                                       cfg.epochs)
            loss, log_vars = model(batch, return_loss=True)

            # meter_dict
            for k, v in log_vars.items():
                if k not in meter_dict.keys():
                    meter_dict[k] = utils.AverageMeter()
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
            if i % 10 == 0:
                log_str = f'Epoch [{epoch}/{cfg.epochs}][{i}/{len(train_loader)}]  '
                log_str += f'lr: {lr:.2g}, eta: {remain_time}, mem: {get_max_memory()}, '\
                    f'data_time: {data_time.val:.2f}, iter_time: {iter_time.val:.2f}'
                for k, v in meter_dict.items():
                    log_str += f', {k}: {v.val:.4f}'
                logger.info(log_str)
        utils.checkpoint_save(epoch, model, optimizer, cfg.work_dir, cfg.save_freq)
