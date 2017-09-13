from .. import config_base as cb
from ..errors import ConfigError


#Types:
BEGAN = 'began'
SCALED_BEGAN_GMSM = 'scaled_began_gmsm'


def config(type_):
    config = cb.Config()

    if type_ == BEGAN:
        config.l1weight = 1.0
        config.gmsweight = 0.0
        config.chromweight = 0.0
    elif type_ == SCALED_BEGAN_GMSM:
        config.l1weight = 2.0
        config.gmsweight = 1.0
        config.chromweight = 0.0
    else:
        raise ConfigError('Invalid config type: {}.'.format(type_))

    config.totalweight = config.l1weight + \
                         config.gmsweight + \
                         config.chromweight

    config.name = type_

    return config
