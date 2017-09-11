from .. import config_base as cb
import numpy as np


def config():
    config = cb.Config()

    config.z_num = 128
    config.hidden_num = 128
    config.output_num = 3
    config.size = 64
    config.repeat_num = int(np.log2(config.size)) - 2
    config.data_format = 'NCHW'
    config.reuse = False

    return config
