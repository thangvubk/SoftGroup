import torch, glob, os, numpy as np
import sys
sys.path.append('../')
from math import cos, pi
from util.log import logger
import os.path as osp
from collections import OrderedDict

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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
        lr =  clip + 0.5 * (base_lr - clip) * \
            (1 + cos(pi * ( (epoch - step_epoch) / (total_epochs - step_epoch)))) 

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))  # area_intersection: K, indicates the number of members in each class in intersection
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def checkpoint_restore(cfg, model, optimizer, exp_path, exp_name, use_cuda=True, epoch=0, dist=False, f=''):
    if use_cuda:
        model.cpu()
    if not f:
        if epoch > 0:
            f = os.path.join(exp_path, exp_name + '-%09d'%epoch + '.pth')
            assert os.path.isfile(f)
        else:
            f = sorted(glob.glob(os.path.join(exp_path, exp_name + '-*.pth')))
            if len(f) > 0:
                f = f[-1]
                epoch = int(f[len(exp_path) + len(exp_name) + 2 : -4])

    if len(f) > 0:
        logger.info('Restore from ' + f)
        checkpoint = torch.load(f)

    
        if 'net' in checkpoint.keys() and 'optimizer' in checkpoint.keys():
            net_checkpoint = checkpoint['net']
            optimizer_checkpoint = checkpoint['optimizer']

            #load net
            for k, v in net_checkpoint.items():
                if 'module.' in k:
                    net_checkpoint = {k[len('module.'):]: v for k, v in net_checkpoint.items()}
                break
            if dist:
                model.module.load_state_dict(net_checkpoint)
            else:
                model.load_state_dict(net_checkpoint)

            # load optimizer
            load_optimizer = getattr(cfg, 'load_optimizer', True)
            if optimizer is not None and load_optimizer == True:
                optimizer.load_state_dict(optimizer_checkpoint)
                for k in optimizer.state.keys():
                    optimizer.state[k]['exp_avg'] = optimizer.state[k]['exp_avg'].cuda()
                    optimizer.state[k]['exp_avg_sq'] = optimizer.state[k]['exp_avg_sq'].cuda()
            
        else: # deprecated without optimizer 
            for k, v in checkpoint.items():
                if 'module.' in k:
                    checkpoint = {k[len('module.'):]: v for k, v in checkpoint.items()}
                break
            if dist:
                model.module.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)

    if use_cuda:
        model.cuda()

    return epoch + 1


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

def checkpoint_save(epoch, model, optimizer, work_dir, save_freq=16):
    f = os.path.join(work_dir, f'epoch_{epoch}.pth')
    checkpoint = {'net': weights_to_cpu(model.state_dict()),
                  'optimizer': optimizer.state_dict(),
                  'epoch': epoch}
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
    state_dict = torch.load(checkpoint)
    src_state_dict = state_dict['net']
    target_state_dict = model.state_dict()
    skip_keys = []
    error_msg = ''
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
        logger.info(f'removed keys in source state_dict due to size mismatch: {", ".join(skip_keys)}')
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

def load_model_param(model, pretrained_dict, prefix=""):
    # suppose every param in model should exist in pretrain_dict, but may differ in the prefix of the name
    # For example:    model_dict: "0.conv.weight"     pretrain_dict: "FC_layer.0.conv.weight"
    model_dict = model.state_dict()
    len_prefix = 0 if len(prefix) == 0 else len(prefix) + 1
    pretrained_dict_filter = {k[len_prefix:]: v for k, v in pretrained_dict.items() if k[len_prefix:] in model_dict and prefix in k}
    assert len(pretrained_dict_filter) > 0
    model_dict.update(pretrained_dict_filter)
    model.load_state_dict(model_dict)
    return len(pretrained_dict_filter), len(model_dict)


def write_obj(points, colors, out_filename):
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        c = colors[i]
        fout.write('v %f %f %f %d %d %d\n' % (points[i,0],points[i,1],points[i,2],c[0],c[1],c[2]))
    fout.close()


def get_batch_offsets(batch_idxs, bs):
    '''
    :param batch_idxs: (N), int
    :param bs: int
    :return: batch_offsets: (bs + 1)
    '''
    batch_offsets = torch.zeros(bs + 1).int().cuda()
    for i in range(bs):
        batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
    assert batch_offsets[-1] == batch_idxs.shape[0]
    return batch_offsets


def print_error(message, user_fault=False):
    sys.stderr.write('ERROR: ' + str(message) + '\n')
    if user_fault:
      sys.exit(2)
    sys.exit(-1)

def get_max_memory():
    mem = torch.cuda.max_memory_allocated()
    mem_mb = torch.tensor([int(mem) // (1024 * 1024)],
                          dtype=torch.int)
    return mem_mb.item()
