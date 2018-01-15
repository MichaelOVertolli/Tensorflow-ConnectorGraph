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
from ..models import leaky_relu
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

#TYPES:
Z128_SZ32 = 'z128_sz32'
Z128_SZ64 = 'z128_sz64'
Z256_SZ64 = 'z256_sz64'
Z1024_SZ64 = 'z1024_sz64'
Z2048_SZ64 = 'z2048_sz64'
Z256_SZ128 = 'z256_sz128'


def config(type_):
    config = cb.Config()

    config.output_num = 3 #channels
    config.data_format = 'NCHW'

    if Z128_SZ32 in type_:
        config.z_num = 128
        config.size = 32
    elif Z128_SZ64 in type_:
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
        raise ConfigError('Invalid z_num or size in type: {}.'.format(type_))

    config.repeat_num = int(np.log2(config.size)) - 1
    config.name = type_

    return config