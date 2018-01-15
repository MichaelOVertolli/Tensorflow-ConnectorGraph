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
    if config.net_name == 'B_im':
        channels = config.hidden_num
    else:
        channels = config.output_num
    B_in = tf.placeholder(tf.float32, [None, channels,
                                       None, None],
                          name='input')
    with tf.variable_scope(config.net_name):
        x = models.slim.conv2d(B_in, config.hidden_num, 3, 1,
                               activation_fn=config.activation_fn,  
                               weights_initializer=config.weights_init,
                               normalizer_fn=config.normalizer_fn,
                               normalizer_params=config.normalizer_params,
                               data_format=config.data_format)
        if config.net_name == 'B_im':
            x = models.slim.conv2d(x, 3, 3, 1, activation_fn=None,
                                   data_format=config.data_format)
    B_out = tf.identity(x, name='output')
    tf.add_to_collection('inputs', B_in)
    tf.add_to_collection('outputs', B_out)

    if config.clone:
        B_in2 = tf.placeholder(tf.float32, [None, channels,
                                            None, None],
                               name='input2')
        with tf.variable_scope(config.net_name, reuse=True):
            x2 = models.slim.conv2d(B_in2, config.hidden_num, 3, 1,
                                    activation_fn=config.activation_fn,  
                                    weights_initializer=config.weights_init,
                                    normalizer_fn=config.normalizer_fn,
                                    normalizer_params=config.normalizer_params,
                                    data_format=config.data_format)
            if config.net_name == 'B_im':
                x2 = models.slim.conv2d(x2, 3, 3, 1, activation_fn=None,
                                        data_format=config.data_format)
        B_out2 = tf.identity(x2, name='output2')
        tf.add_to_collection('inputs', B_in2)
        tf.add_to_collection('outputs', B_out2)
    
    saver = tf.train.Saver()
    return saver



