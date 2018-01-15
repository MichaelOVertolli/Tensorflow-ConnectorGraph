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
    G_in = tf.placeholder(tf.float32, [None, config.z_num], name='input')
    tf.add_to_collection('inputs', G_in)
    if config.block == 0:
        with tf.variable_scope('front') as vs:
            fc_size = np.prod([config.hidden_num, config.size, config.size])
            G_in = models.slim.fully_connected(G_in, fc_size, activation_fn=None)
            G_in = models.reshape(G_in, config.size, config.size,
                                  config.hidden_num, config.data_format)
    if config.alpha:
        alpha = tf.placeholder(tf.float32, (), name='alpha'+str(config.block))
        tf.add_to_collection('inputs', alpha)
        alpha_pair = {'residual': alpha, 'shortcut': 1-alpha}
    else:
        alpha_pair = {'residual': 1.0, 'shortcut': 1.0}
    G_out, G_all_vars = models.ResidualBlock(G_in,
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
    G_out = tf.identity(G_out, name='output')
    tf.add_to_collection('outputs', G_out)

    if config.clone:
        G2_in = tf.placeholder(tf.float32, [None, config.z_num], name='input2')
        tf.add_to_collection('inputs', G2_in)
        if config.block == 0:
            with tf.variable_scope('front', reuse=True):
                fc_size = np.prod([config.hidden_num, config.size, config.size])
                G2_in = models.slim.fully_connected(G2_in, fc_size, activation_fn=None)
                G2_in = models.reshape(G2_in, config.size, config.size,
                                       config.hidden_num, config.data_format)
        G2_out, G2_all_vars = models.ResidualBlock(G2_in,
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
                                                   True,
                                                   config.data_format)
        G2_out = tf.identity(G2_out, name='output2')
        tf.add_to_collection('outputs', G2_out)
    
    saver = tf.train.Saver()
    return saver



