import logging

from tensorboardX import SummaryWriter as _SummaryWriter

from .dist import is_main_process, master_only


def get_root_logger(log_file=None, log_level=logging.INFO):
    logger = logging.getLogger('softgroup')
    # if the logger has been initialized, just return it
    if logger.hasHandlers():
        return logger

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=log_level)
    if not is_main_process():
        logger.setLevel('ERROR')
    elif log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    return logger


class SummaryWriter(_SummaryWriter):

    @master_only
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

    @master_only
    def add_scalar(self, *args, **kwargs):
        return super().add_scalar(*args, **kwargs)

    @master_only
    def flush(self, *args, **kwargs):
        return super().flush(*args, **kwargs)
