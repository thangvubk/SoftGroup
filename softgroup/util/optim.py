import torch.optim


def build_optimizer(model, optim_cfg):
    assert 'type' in optim_cfg
    _optim_cfg = optim_cfg.copy()
    optim_type = _optim_cfg.pop('type')
    optim = getattr(torch.optim, optim_type)
    return optim(filter(lambda p: p.requires_grad, model.parameters()), **_optim_cfg)
