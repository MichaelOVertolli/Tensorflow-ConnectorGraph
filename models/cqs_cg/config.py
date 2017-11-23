###############################################################################
#Copyright (C) 2017  Michael O. Vertolli michaelvertolli@gmail.com
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program.  If not, see http://www.gnu.org/licenses/
###############################################################################

from .. import config_base as cb
from ..errors import ConfigError

#batch_size
B_16 = 'b16'
#gamma
G_5 = 'g0.5'
G_7 = 'g0.7'
#lss_type
BEGAN = 'began'
SCALED_BEGAN_GMSM = 'scaled_began_gmsm'
#mdl_type
Z128_SZ64 = 'z128_sz64'
Z256_SZ64 = 'z256_sz64'
Z1024_SZ64 = 'z1024_sz64'
Z2048_SZ64 = 'z2048_sz64'
Z256_SZ128 = 'z256_sz128'
H128 = 'h128'
H256 = 'h256'


def config(type_):
    config = cb.Config()


    if B_16 in type_:
        config.batch_size = 16
    else:
        raise ConfigError('Invalid config type {} for batch_size.'.format(type_))
    if G_5 in type_:
        config.gamma = 0.5
    elif G_7 in type_:
        config.gamma = 0.7
    else:
        raise ConfigError('Invalid config type {} for gamma.'.format(type_))
    if SCALED_BEGAN_GMSM in type_:
        config.lss_type = SCALED_BEGAN_GMSM
    elif BEGAN in type_:
        config.lss_type = BEGAN
    else:
        raise ConfigError('Invalid config type {} for lss_type.'.format(type_))
    if Z128_SZ64 in type_:
        config.mdl_type = Z128_SZ64
        config.z_num = 128
        config.img_size = 64
    elif Z256_SZ64 in type_:
        config.mdl_type = Z256_SZ64
        config.z_num = 256
        config.img_size = 64
    elif Z1024_SZ64 in type_:
        config.mdl_type = Z1024_SZ64
        config.z_num = 1024
        config.img_size = 64
    elif Z2048_SZ64 in type_:
        config.mdl_type = Z2048_SZ64
        config.z_num = 2048
        config.img_size = 64
    elif Z256_SZ128 in type_:
        config.mdl_type = Z256_SZ128
        config.z_num = 256
        config.img_size = 128
    else:
        raise ConfigError('Invalid config type {} for mdl_type.'.format(type_))
    if H128 in type_:
        config.mdl_type = '_'.join([config.mdl_type, H128])
    elif H256 in type_:
        config.mdl_type = '_'.join([config.mdl_type, H256])
    else:
        raise ConfigError('Invalid config type: {}.'.format(type_))
    
    config.g_lr = 8e-5
    config.d_lr = 8e-5
    config.lr_lower_boundary = 2e-5
    config.lambda_k = 0.001

    config.data_format = 'NCHW'

    config.name = type_

    return config
