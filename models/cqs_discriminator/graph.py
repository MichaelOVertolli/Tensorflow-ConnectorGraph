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


def build_graph(model_name, config):
    D_in = tf.placeholder(tf.float32, [None, config.output_num, config.size, config.size], name='input')
    D_out, D_z, D_vars = models.DiscriminatorCNN(D_in,
                                                 config.output_num, config.z_num,
                                                 config.repeat_num, config.hidden_num,
                                                 config.data_format)
    D_out = tf.identity(D_out, name='output') 
    tf.add_to_collection('inputs', D_in)
    tf.add_to_collection('outputs', D_out)
    #tf.add_to_collection('trainable', G_vars)
    saver = tf.train.Saver()
    return saver



