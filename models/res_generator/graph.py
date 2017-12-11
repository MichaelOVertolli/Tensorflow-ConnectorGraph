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
    if config.alpha:
        alphas = [tf.placeholder(tf.float32, (), name='alpha'+str(i)) for i in range(config.repeat_num-1)]
        for alpha in alphas:
            tf.add_to_collection('inputs', alpha)
    else:
        alphas = None
    G_out, G_all_vars, G_sep_vars = models.ResNet(G_in,
                                                  config.repeat_num,
                                                  config.hidden_nums,
                                                  config.sizes,
                                                  config.activation_fn,
                                                  config.net_name,
                                                  alphas,
                                                  config.project,
                                                  config.minibatch,
                                                  config.resample,
                                                  config.pool_type,
                                                  config.weights_init,
                                                  config.normalizer_fn,
                                                  config.normalizer_params,
                                                  False,
                                                  config.data_format)
    if config.tanh:
        G_out = tf.tanh(G_out, name='output')
    else:
        G_out = tf.identity(G_out, name='output')
    tf.add_to_collection('inputs', G_in)
    tf.add_to_collection('outputs', G_out)
    for key in G_sep_vars:
        col = '_'.join([key, 'variables'])
        for var in G_sep_vars[key]:
            tf.add_to_collection(col, var)
    #tf.add_to_collection('trainable', G_vars)
    saver = tf.train.Saver()
    return saver



