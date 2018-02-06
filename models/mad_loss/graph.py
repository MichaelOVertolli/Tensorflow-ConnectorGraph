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
    
    inpt = tf.placeholder(tf.float32, [None, None, None, None], name='input')
    tf.add_to_collection('inputs', inpt)
    inpt = inpt[config.batch_size:, :, :, :] # strips real_input
    inpt_bs = config.count * config.batch_size
    indices = range(0, inpt_bs+1, config.batch_size)
    indices = zip(indices[:-1], indices[1:])
    outputs = []
    for i, j in indices:
        x = tf.tile(inpt[i:j, :, :, :], [config.count, 1, 1, 1])
        x = tf.reduce_sum(tf.abs(x - inpt), 0)/(inpt_bs - config.batch_size) # collapses out the zeros where x = x
        x = tf.reduce_mean(x)/2.0 # re-normalize to [0, 1]
        x = 1.0 - x # identity is bad for diversity so flip value 
        outputs.append(x)
    make_null_variable()
    for i, output in enumerate(outputs):
        o = tf.identity(output, name='output{}'.format(i))
        tf.add_to_collection('outputs', o)
    saver = tf.train.Saver()
    return saver



