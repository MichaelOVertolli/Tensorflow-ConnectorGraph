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
    
    orig = tf.placeholder(tf.float32, [None, None, None, None],
                          name='orig_input')
    autoencoded = tf.placeholder(tf.float32, [None, None, None, None],
                                 name='autoencoded_input')
    l1 = tf.reduce_mean(tf.abs(autoencoded - orig))
    gms, chrom = friqa.prep_and_call_qs(orig, autoencoded)
    loss = (config.l1weight*l1 +
            config.gmsweight*gms +
            config.chromweight*chrom) / config.totalweight
    loss = tf.identity(loss, name='output')
    make_null_variable()
    tf.add_to_collection('inputs', orig)
    tf.add_to_collection('inputs', autoencoded)
    tf.add_to_collection('outputs', loss)
    saver = tf.train.Saver()
    return saver



