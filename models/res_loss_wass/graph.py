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
from .. import friqa
from ..null_variable import make_null_variable


def build_graph(config):
    
    inpt = tf.placeholder(tf.float32, [None, config.z_num],
                          name='input')
    mix_in = tf.placeholder(tf.float32, [None, None, None, None],
                         name='mix_input')

    with tf.variable_scope('D_linear'):
        x = models.slim.fully_connected(inpt, 1, activation_fn=None)

    if config.improved:
        data, mix, gen = tf.split(x, 3)
    else:
        data, gen = tf.split(x, 2)

    g_loss = -tf.reduce_mean(gen)
    d_loss = tf.reduce_mean(gen) - tf.reduce_mean(data)

    if config.improved:
        mix_grads = tf.gradients(tf.reduce_sum(mix), mix_in)[0]
        mix_norms = tf.sqrt(tf.reduce_sum(mix_grads**2, [1, 2, 3]))
        grad_penalty = tf.reduce_mean((mix_norms - config.iwass_target)**2) * config.iwass_lambda / (config.iwass_target**2)            
        d_loss += grad_penalty
        if config.nvd_grad:
            nvd_penalty = tf.reduce_mean(data**2) * config.iwass_epsilon
            d_loss += nvd_penalty

    g_loss = tf.identity(g_loss, name='g_loss')
    d_loss = tf.identity(d_loss, name='d_loss')

    tf.add_to_collection('inputs', inpt)
    tf.add_to_collection('inputs', mix)
    tf.add_to_collection('outputs', g_loss)
    tf.add_to_collection('outputs', d_loss)
    saver = tf.train.Saver()
    return saver



