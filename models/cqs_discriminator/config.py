from .. import config_base as cb
from ..errors import ConfigError
import numpy as np


#TYPES:
Z128_SZ64 = 'z128_sz64'
Z256_SZ64 = 'z256_sz64'
Z1024_SZ64 = 'z1024_sz64'
Z2048_SZ64 = 'z2048_sz64'
Z256_SZ128 = 'z256_sz128'
H128 = 'h128'
H256 = 'h256'


def config(type_):
    config = cb.Config()

    config.output_num = 3 #channels
    config.data_format = 'NCHW'

    if Z128_SZ64 in type_:
        config.z_num = 128
        config.size = 64
    elif Z256_SZ64 in type_:
        config.z_num = 256
        config.size = 64
    elif Z1024_SZ64 in type_:
        config.z_num = 1024
        config.size = 64
    elif Z2048_SZ64 in type_:
        config.z_num = 2048
        config.size = 64
    elif Z256_SZ128 in type_:
        config.z_num = 256
        config.size = 128
    else:
        raise ConfigError('Invalid config type: {}.'.format(type_))
    if H128 in type_:
        config.hidden_num = 128
    elif H256 in type_:
        config.hidden_num = 256
    else:
        raise ConfigError('Invalid config type: {}.'.format(type_))

    config.repeat_num = int(np.log2(config.size)) - 2
    config.name = type_

    return config
