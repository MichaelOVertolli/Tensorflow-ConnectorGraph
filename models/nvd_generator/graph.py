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
    # alphas = [tf.Variable(0.1, trainable=False, name='alpha'+str(i), dtype=tf.float32)
    #           for i in range(config.repeat_num-1)]
    alphas = [tf.placeholder(tf.float32, (), name='alpha'+str(i)) for i in range(config.repeat_num-1)]
    G_in = tf.placeholder(tf.float32, [None, config.z_num], name='input')
    G_outs, G_vars = models.GeneratorSkipCNN(G_in,
                                             config.hidden_num, config.output_num,
                                             config.repeat_num, alphas,
                                             config.data_format, False)
    for alpha in alphas:
        tf.add_to_collection('inputs', alpha)
    tf.add_to_collection('inputs', G_in)
    for i in range(len(G_outs)):
        temp = tf.identity(G_outs[i], name='output'+str(i))
        tf.add_to_collection('outputs', temp)
    saver = tf.train.Saver()
    return saver



