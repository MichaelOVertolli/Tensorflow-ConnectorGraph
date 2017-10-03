from models.config_base import Config
from models.errors import ConfigError


def config():
    config = Config()

    config.max_step = 100
    config.start_step = 0
    config.save_step = 5000
    config.log_step = 50

    config.batch_size = 16

    return config

