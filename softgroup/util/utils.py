import functools
import os
import os.path as osp
from collections import OrderedDict
from math import cos, pi

import torch
from torch import distributed as dist

from .dist import get_dist_info, master_only


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self, apply_dist_reduce=False):
        self.apply_dist_reduce = apply_dist_reduce
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def dist_reduce(self, val):
        rank, world_size = get_dist_info()
        if world_size == 1:
            return val
        if not isinstance(val, torch.Tensor):
            val = torch.tensor(val, device='cuda')
        dist.all_reduce(val)
        return val.item() / world_size

    def get_val(self):
        if self.apply_dist_reduce:
            return self.dist_reduce(self.val)
        else:
            return self.val

    def get_avg(self):
        if self.apply_dist_reduce:
            return self.dist_reduce(self.avg)
        else:
            return self.avg

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Epoch counts from 0 to N-1
def cosine_lr_after_step(optimizer, base_lr, epoch, step_epoch, total_epochs, clip=1e-6):
    if epoch < step_epoch:
        lr = base_lr
    else:
        lr = clip + 0.5 * (base_lr - clip) * \
            (1 + cos(pi * ((epoch - step_epoch) / (total_epochs - step_epoch))))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def is_power2(num):
    return num != 0 and ((num & (num - 1)) == 0)


def is_multiple(num, multiple):
    return num != 0 and num % multiple == 0


def weights_to_cpu(state_dict):
    """Copy a model state_dict to cpu.

    Args:
        state_dict (OrderedDict): Model weights on GPU.
    Returns:
        OrderedDict: Model weights on GPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    return state_dict_cpu


@master_only
def checkpoint_save(epoch, model, optimizer, work_dir, save_freq=16):
    if hasattr(model, 'module'):
        model = model.module
    f = os.path.join(work_dir, f'epoch_{epoch}.pth')
    checkpoint = {
        'net': weights_to_cpu(model.state_dict()),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, f)
    if os.path.exists(f'{work_dir}/latest.pth'):
        os.remove(f'{work_dir}/latest.pth')
    os.system(f'cd {work_dir}; ln -s {osp.basename(f)} latest.pth')

    # remove previous checkpoints unless they are a power of 2 or a multiple of save_freq
    epoch = epoch - 1
    f = os.path.join(work_dir, f'epoch_{epoch}.pth')
    if os.path.isfile(f):
        if not is_multiple(epoch, save_freq) and not is_power2(epoch):
            os.remove(f)


def load_checkpoint(checkpoint, logger, model, optimizer=None, strict=False):
    if hasattr(model, 'module'):
        model = model.module
    device = torch.cuda.current_device()
    state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage.cuda(device))
    src_state_dict = state_dict['net']
    target_state_dict = model.state_dict()
    skip_keys = []
    # skip mismatch size tensors in case of pretraining
    for k in src_state_dict.keys():
        if k not in target_state_dict:
            continue
        if src_state_dict[k].size() != target_state_dict[k].size():
            skip_keys.append(k)
    for k in skip_keys:
        del src_state_dict[k]
    missing_keys, unexpected_keys = model.load_state_dict(src_state_dict, strict=strict)
    if skip_keys:
        logger.info(
            f'removed keys in source state_dict due to size mismatch: {", ".join(skip_keys)}')
    if missing_keys:
        logger.info(f'missing keys in source state_dict: {", ".join(missing_keys)}')
    if unexpected_keys:
        logger.info(f'unexpected key in source state_dict: {", ".join(unexpected_keys)}')

    # load optimizer
    if optimizer is not None:
        assert 'optimizer' in state_dict
        optimizer.load_state_dict(state_dict['optimizer'])

    if 'epoch' in state_dict:
        epoch = state_dict['epoch']
    else:
        epoch = 0
    return epoch + 1


def get_max_memory():
    mem = torch.cuda.max_memory_allocated()
    mem_mb = torch.tensor([int(mem) // (1024 * 1024)], dtype=torch.int, device='cuda')
    _, world_size = get_dist_info()
    if world_size > 1:
        dist.reduce(mem_mb, 0, op=dist.ReduceOp.MAX)
    return mem_mb.item()


def cuda_cast(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        new_args = []
        for x in args:
            if isinstance(x, torch.Tensor):
                x = x.cuda()
            new_args.append(x)
        new_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.cuda()
            new_kwargs[k] = v
        return func(*new_args, **new_kwargs)

    return wrapper
