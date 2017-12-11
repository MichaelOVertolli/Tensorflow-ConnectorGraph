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
import numpy as np


#TYPES:
Z128_SZ64 = 'z128_sz64'
Z256_SZ64 = 'z256_sz64'
Z512_SZ64 = 'z512_sz64'
Z1024_SZ64 = 'z1024_sz64'
Z2048_SZ64 = 'z2048_sz64'
Z256_SZ128 = 'z256_sz128'
Z512_SZ64 = 'z512_sz64'
Z512_SZ256 = 'z512_sz256'
Z512_SZ512 = 'z512_sz512'
Z512_SZ1024 = 'z512_sz1024'
H128 = 'h128'
H256 = 'h256'
H512 = 'h512'

def config(type_):
    config = cb.Config()
    
    config.output_num = 3 #channels
    config.data_format = 'NCHW'
    config.base_size = 4

    if Z128_SZ64 in type_:
        config.z_num = 128
        config.size = 64
    elif Z256_SZ64 in type_:
        config.z_num = 256
        config.size = 64
    elif Z512_SZ64 in type_:
        config.z_num = 512
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
    elif Z512_SZ64 in type_:
        config.z_num = 512
        config.size = 64
    elif Z512_SZ256 in type_:
        config.z_num = 512
        config.size = 256
    elif Z512_SZ512 in type_:
        config.z_num = 512
        config.size = 512
    elif Z512_SZ1024 in type_:
        config.z_num = 512
        config.size = 1024
    else:
        raise ConfigError('Invalid config type: {}.'.format(type_))
    if H128 in type_:
        config.hidden_num = 128
    elif H256 in type_:
        config.hidden_num = 256
    elif H512 in type_:
        config.hidden_num = 512
    else:
        raise ConfigError('Invalid config type: {}.'.format(type_))

    config.max_dist = 12 #needs to be changed to incr/decr filter banks
    config.repeat_num = int(np.log2(config.size)) - 1
    config.name = type_

    return config
