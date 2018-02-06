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
import re
import tensorflow as tf
slim = tf.contrib.slim

#TYPES:
Z128_SZ32 = 'z128_sz32'
Z256_SZ32 = 'z256_sz32'
Z128_SZ64 = 'z128_sz64'
Z256_SZ64 = 'z256_sz64'
Z1024_SZ64 = 'z1024_sz64'
Z2048_SZ64 = 'z2048_sz64'
Z256_SZ128 = 'z256_sz128'
H128 = 'h128'
H256 = 'h256'
H512 = 'h512'
PROJ = 'project'
ALPH = 'alphas'
ELU_ = 'elu'
RELU = 'relu'
LRLU = 'leaky_relu'
PCNV = 'pconv'
PAVG = 'pavg'
WXAV = 'wxav'
WHE_ = 'whe'
NPIX = 'npixel'
NBAT = 'nbatch'
MINI = 'minibatch'
TANH = 'tanh'
BLCK = 'block'
CLON = 'clone'
BASE = 'base'


def config(type_):
    config = cb.Config()

    m = re.search('(?<={})\d+'.format(BLCK), type_)
    if m is not None:
        config.block = int(m.group(0))
    else:
        raise ConfigError('Invalid block in type: {}.'.format(type_))
    
    config.net_name = 'G'
    config.output_num = 3 #channels
    config.data_format = 'NCHW'
    config.resample = 'up'

    if Z128_SZ32 in type_:
        config.z_num = 128
        config.size = 32
    elif Z256_SZ32 in type_:
        config.z_num = 256
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
    if H128 in type_:
        config.hidden_num = 128
    elif H256 in type_:
        config.hidden_num = 256
    elif H512 in type_:
        config.hidden_num = 512
    else:
        raise ConfigError('Invalid hidden_num in type: {}.'.format(type_))
    if LRLU in type_:
        config.activation_fn = leaky_relu
    elif RELU in type_:
        config.activation_fn = tf.nn.relu
    elif ELU_ in type_:
        config.activation_fn = tf.nn.elu
    else:
        raise ConfigError('Invalid activation_fn in type: {}.'.format(type_))
    if PCNV in type_:
        config.pool_type = 'conv'
    elif PAVG in type_:
        config.pool_type = 'avg_pool'
    else:
        raise ConfigError('Invalid pool_type in type: {}.'.format(type_))
    if WXAV in type_:
        config.weights_init = tf.contrib.layers.xavier_initializer()
    elif WHE_ in type_:
        config.weights_init = tf.contrib.layers.variance_scaling_initializer()
    else:
        raise ConfigError('Invalid weights_init in type: {}.'.format(type_))
    if NPIX in type_:
        config.normalizer_fn = slim.unit_norm
        config.normalizer_params = {'dim':1, 'epsilon':1e-8}
    elif NBAT in type_:
        config.normalizer_fn = slim.batch_norm
        config.normalizer_params = None
    else:
        config.normalizer_fn = None
        config.normalizer_params = None
    if TANH in type_:
        config.tanh = True
    else:
        config.tanh = False
    if CLON in type_:
        config.clone = True
    else:
        config.clone = False

    config.repeat_num = int(np.log2(config.size)) - 1

    if PROJ in type_:
        config.project = True
        max_ = config.repeat_num + 2
        hidden_nums = [min(2**(max_-i), config.hidden_num) for i in range(config.repeat_num)]
        config.hidden_num = hidden_nums[config.block]
    else:
        config.project = False
    if ALPH in type_:
        config.alpha = True
    else:
        config.alpha = False
    if MINI in type_:
        config.minibatch = True
    else:
        config.minibatch = False
    if BASE in type_:
        config.base = True
    else:
        config.base = False

    sizes = [2**i for i in range(2, int(np.log2(config.size)))]
    config.size = sizes[config.block]
    config.name = type_

    return config
