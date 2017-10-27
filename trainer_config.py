from models.config_base import Config
from models.errors import ConfigError


def config():
    config = Config()

    config.use_gpu = True

    config.data_format = 'NCHW'

    config.max_step = 200000
    config.start_step = 0
    config.save_step = 5000
    config.log_step = 50
    config.lr_update_step = 100000

    return config

