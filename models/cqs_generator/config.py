from .. import config_base as cb
from ..errors import ConfigError
import numpy as np


#TYPES:
Z128_SZ64 = 'z128_sz64'
Z128_SZ64_REUSE = 'z128_sz64_reuse'
Z256_SZ128 = 'z256_sz128'


def config(type_):
    config = cb.Config()

    config.hidden_num = 128
    config.output_num = 3 #channels
    config.data_format = 'NCHW'
    config.batch = 16

    if type_ == Z128_SZ64:
        config.z_num = 128
        config.size = 64
        config.repeat_num = int(np.log2(config.size)) - 2
        config.reuse = False
    elif type_ == Z128_SZ64_REUSE:
        config.z_num = 128
        config.size = 64
        config.repeat_num = int(np.log2(config.size)) - 2
        config.reuse = True
    elif type_ == Z256_SZ128:
        config.z_num = 256
        config.size = 128
        config.repeat_num = int(np.log2(config.size)) - 2
        config.reuse = False
    else:
        raise ConfigError('Invalid config type: {}.'.format(type_))
    
    config.name = type_

    return config
