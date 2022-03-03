import logging
import os
import sys
import time

sys.path.append('../')

from util.config import cfg


def create_logger(log_file):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG if cfg.local_rank == 0 else 'ERROR')

    handler = logging.StreamHandler()
    log_format = '[%(asctime)s  %(levelname)s  %(filename)s  line %(lineno)d  %(process)d]  %(message)s'
    handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(handler)

    logging.basicConfig(level=logging.DEBUG, format=log_format, filename=log_file)    # filename: build a FileHandler
    return logger


if cfg.task == 'train':
    log_file = os.path.join(
        cfg.exp_path,
        'train-{}.log'.format(time.strftime("%Y%m%d_%H%M%S", time.localtime()))
    )
elif cfg.task == 'test':
    log_file = os.path.join(
        cfg.exp_path, 'result', 'epoch{}_nmst{}_scoret{}_npointt{}'.format(cfg.test_epoch, cfg.TEST_NMS_THRESH, cfg.TEST_SCORE_THRESH, cfg.TEST_NPOINT_THRESH),
        cfg.split, 'test-{}.log'.format(time.strftime("%Y%m%d_%H%M%S", time.localtime()))
    )
if not os.path.exists(os.path.dirname(log_file)):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)


logger = create_logger(log_file)
logger.info('************************ Start Logging ************************')