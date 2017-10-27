from .. import config_base as cb
from ..errors import ConfigError


G_5 = 'g_0.5'
G_7 = 'g_0.7'


def config(type_):
    config = cb.Config()


    if type_ == G_5:
        config.gamma = 0.5
    elif type_ == G_7:
        config.gamma = 0.7
    else:
        raise ConfigError('Invalid config type: {}.'.format(type_))
    
    config.g_lr = 2e-5
    config.d_lr = 8e-5
    config.lr_lower_boundary = 2e-5
    config.lambda_k = 0.001

    config.data_format = 'NCHW'

    config.name = type_

    return config
