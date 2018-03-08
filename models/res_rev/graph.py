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

import numpy as np
import tensorflow as tf
from .. import models


def build_graph(config):
    R_in = tf.placeholder(tf.float32, [None, config.hidden_num,
                                       config.size, config.size],
                          name='input')
    if config.alpha:
        alpha = tf.placeholder(tf.float32, (), name='alpha'+str(config.block))
        tf.add_to_collection('inputs', alpha)
        if config.avariant:
            alpha_pair = {'residual': alpha, 'shortcut': 1.0}
        else:
            alpha_pair = {'residual': alpha, 'shortcut': 1-alpha}
    else:
        alpha_pair = {'residual': 1.0, 'shortcut': 1.0}
    R_out, R_vars = models.ResidualBlock(R_in,
                                         config.hidden_num,
                                         config.size,
                                         config.activation_fn,
                                         config.net_name,
                                         alpha_pair,
                                         config.project,
                                         config.resample,
                                         config.pool_type,
                                         config.weights_init,
                                         config.normalizer_fn,
                                         config.normalizer_params,
                                         False,
                                         config.data_format)
                                        
    if config.base:
        with tf.variable_scope('end'):
            R_out = models.slim.conv2d(R_out, config.z_num, 4, 1, padding='VALID',
                                       activation_fn=config.activation_fn, 
                                       weights_initializer=config.weights_init,
                                       normalizer_fn=config.normalizer_fn,
                                       normalizer_params=config.normalizer_params,
                                       data_format=config.data_format)
            R_out = tf.squeeze(R_out, [2, 3])
            
    R_out = tf.identity(R_out, name='output')
    tf.add_to_collection('inputs', R_in)
    tf.add_to_collection('outputs', R_out)
    
    saver = tf.train.Saver()
    return saver



