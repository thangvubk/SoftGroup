import torch
import torch.optim as optim
import time, sys, os, random
from tensorboardX import SummaryWriter
import numpy as np

from util.config import cfg

import torch.distributed as dist


def init():
    # copy important files to backup
    backup_dir = os.path.join(cfg.exp_path, 'backup_files')
    os.makedirs(backup_dir, exist_ok=True)
    os.system('cp train.py {}'.format(backup_dir))
    os.system('cp {} {}'.format(cfg.model_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.dataset_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.config, backup_dir))

    # log the config
    logger.info(cfg)

    # summary writer
    global writer
    writer = SummaryWriter(cfg.exp_path)

    # random seed
    random.seed(cfg.manual_seed)
    np.random.seed(cfg.manual_seed)
    torch.manual_seed(cfg.manual_seed)
    torch.cuda.manual_seed_all(cfg.manual_seed)

# epoch counts from 1 to N
def train_epoch(train_loader, model, model_fn, optimizer, epoch):
    iter_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    am_dict = {}

    model.train()
    start_epoch = time.time()
    end = time.time()

    if train_loader.sampler is not None and cfg.dist == True:
        train_loader.sampler.set_epoch(epoch)

    for i, batch in enumerate(train_loader):

        if batch['locs'].shape[0] < 20000:
            logger.info("point num < 20000, continue")
            continue

        data_time.update(time.time() - end)
        torch.cuda.empty_cache()

        # adjust learning rate
        utils.cosine_lr_after_step(optimizer, cfg.lr, epoch - 1, cfg.step_epoch, cfg.epochs)
    

        # prepare input and forward
        loss, _, visual_dict, meter_dict = model_fn(batch, model, epoch, semantic_only=cfg.semantic_only)

        # meter_dict
        for k, v in meter_dict.items():
            if k not in am_dict.keys():
                am_dict[k] = utils.AverageMeter()
            am_dict[k].update(v[0], v[1])

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # time and print
        current_iter = (epoch - 1) * len(train_loader) + i + 1
        max_iter = cfg.epochs * len(train_loader)
        remain_iter = max_iter - current_iter

        iter_time.update(time.time() - end)
        end = time.time()

        remain_time = remain_iter * iter_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if cfg.local_rank == 0 and i % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            sys.stdout.write(
                "epoch: {}/{} iter: {}/{} lr: {:.5f} loss: {:.4f}({:.4f}) data_time: {:.2f}({:.2f}) iter_time: {:.2f}({:.2f}) remain_time: {remain_time}\n".format
                (epoch, cfg.epochs, i + 1, len(train_loader), lr, am_dict['loss'].val, am_dict['loss'].avg,
                data_time.val, data_time.avg, iter_time.val, iter_time.avg, remain_time=remain_time))
     
        if (i == len(train_loader) - 1): print()


    logger.info("epoch: {}/{}, train loss: {:.4f}, time: {}s".format(epoch, cfg.epochs, am_dict['loss'].avg, time.time() - start_epoch))

    if cfg.local_rank == 0:
        utils.checkpoint_save(model, optimizer, cfg.exp_path, cfg.config.split('/')[-1][:-5], epoch, cfg.save_freq, use_cuda)

    for k in am_dict.keys():
        if k in visual_dict.keys():
            writer.add_scalar(k+'_train', am_dict[k].avg, epoch)


def eval_epoch(val_loader, model, model_fn, epoch):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    am_dict = {}

    with torch.no_grad():
        model.eval()
        start_epoch = time.time()
        for i, batch in enumerate(val_loader):

            # prepare input and forward
            loss, preds, visual_dict, meter_dict = model_fn(batch, model, epoch, semantic_only=cfg.semantic_only)

            for k, v in meter_dict.items():
                if k not in am_dict.keys():
                    am_dict[k] = utils.AverageMeter()
                am_dict[k].update(v[0], v[1])
            sys.stdout.write("\riter: {}/{} loss: {:.4f}({:.4f})".format(i + 1, len(val_loader), am_dict['loss'].val, am_dict['loss'].avg))
            if (i == len(val_loader) - 1): print()

        logger.info("epoch: {}/{}, val loss: {:.4f}, time: {}s".format(epoch, cfg.epochs, am_dict['loss'].avg, time.time() - start_epoch))

        for k in am_dict.keys():
            if k in visual_dict.keys():
                writer.add_scalar(k + '_eval', am_dict[k].avg, epoch)


if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    if cfg.dist == True:
        raise NotImplementedError
        # num_gpus = torch.cuda.device_count()
        # dist.init_process_group(backend='nccl', rank=cfg.local_rank, 
        #     world_size=num_gpus)
        # torch.cuda.set_device(cfg.local_rank)

    from util.log import logger
    import util.utils as utils

    init()

    exp_name = cfg.config.split('/')[-1][:-5]
    model_name = exp_name.split('_')[0]
    data_name = exp_name.split('_')[-1]

    # model
    logger.info('=> creating model ...')
    if model_name == 'softgroup':
        from model.softgroup.softgroup import SoftGroup as Network
        from model.softgroup.softgroup import model_fn_decorator
    else:
        print("Error: no model - " + model_name)
        exit(0)

    model = Network(cfg)

    use_cuda = torch.cuda.is_available()
    logger.info('cuda available: {}'.format(use_cuda))
    assert use_cuda
    model = model.cuda()

    logger.info('#classifier parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))

    # optimizer
    if cfg.optim == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    elif cfg.optim == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
            lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)


    model_fn = model_fn_decorator()

    # dataset
    if cfg.dataset == 'scannetv2':
        if data_name == 'scannet':
            import data.scannetv2_inst
            dataset = data.scannetv2_inst.Dataset()
            if cfg.dist:
                dataset.dist_trainLoader()
            else:
                dataset.trainLoader()
            dataset.valLoader()
        else:
            print("Error: no data loader - " + data_name)
            exit(0)
    elif cfg.dataset == 's3dis' and data_name == 's3dis':
        import data.s3dis_inst
        dataset = data.s3dis_inst.Dataset()
        dataset.trainLoader()
        dataset.valLoader()
    else:
        raise NotImplementedError("Not yet supported")


    # resume from the latest epoch, or specify the epoch to restore
    start_epoch = utils.checkpoint_restore(cfg, model, optimizer, cfg.exp_path, 
        cfg.config.split('/')[-1][:-5], use_cuda)      


    if cfg.dist:
        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
                model.cuda(cfg.local_rank),
                device_ids=[cfg.local_rank],
                output_device=cfg.local_rank,
                find_unused_parameters=True)


    # train and val
    for epoch in range(start_epoch, cfg.epochs + 1):
        train_epoch(dataset.train_data_loader, model, model_fn, optimizer, epoch)

        if utils.is_multiple(epoch, cfg.save_freq) or utils.is_power2(epoch):
            eval_epoch(dataset.val_data_loader, model, model_fn, epoch)
