from models.config_base import Config
from models.errors import ConfigError


def config():
    config = Config()

    config.use_gpu = True

    config.data_dir = './data/CelebA'

    config.max_step = 100
    config.start_step = 0
    config.save_step = 5000
    config.log_step = 50
    config.lr_update_step = 100000

    config.batch_size = 16
    config.img_size = 64
    config.z_num = 128

    return config

