from .s3dis import S3DISDataset
from .scannetv2 import ScanNetDataset
from torch.utils.data import DataLoader

__all__ = ['S3DISDataset', 'ScanNetDataset', 'build_dataset']


def build_dataset(data_cfg, logger):
    assert 'type' in data_cfg
    _data_cfg = data_cfg.copy()
    _data_cfg['logger'] = logger
    data_type = _data_cfg.pop('type')
    if data_type == 's3dis':
        return S3DISDataset(**_data_cfg)
    elif data_type == 'scannetv2':
        return ScanNetDataset(**_data_cfg)
    else:
        raise ValueError(f'Unknown {data_type}')


def build_dataloader(dataset, batch_size=1, num_workers=1, training=True):
    if training:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=dataset.collate_fn,
            shuffle=True,
            drop_last=True,
            pin_memory=True)
    else:
        assert batch_size == 1 and num_workers == 1
        return DataLoader(
            dataset,
            batch_size=1,
            num_workers=1,
            collate_fn=dataset.collate_fn,
            shuffle=False,
            drop_last=False,
            pin_memory=True)
