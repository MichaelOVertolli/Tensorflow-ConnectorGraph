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
    G_in = tf.placeholder(tf.float32, [None, config.z_num], name='G1_input')
    G_out, G_vars = models.GeneratorCNN(G_in,
                                        config.hidden_num, config.output_num,
                                        config.repeat_num, config.data_format,
                                        False)
    G_out = tf.identity(G_out, name='G1_output')
    tf.add_to_collection('inputs', G_in)
    tf.add_to_collection('outputs', G_out)
    for v in G_vars:
        tf.add_to_collection('G_tensors', v)
    GR_in = tf.identity(G_out, name='GR_input')
    GR_out, GR_vars = models.GeneratorRCNN(GR_in,
                                           config.output_num, config.z_num,
                                           config.repeat_num, config.hidden_num,
                                           config.data_format)
    GR_out = tf.identity(GR_out, name='GR_output')
    tf.add_to_collection('inputs', GR_in)
    tf.add_to_collection('outputs', GR_out)
    for v in GR_vars:
        tf.add_to_collection('GR_tensors', v)
    G2_out, G2_vars = models.GeneratorCNN(GR_out,
                                          config.hidden_num, config.output_num,
                                          config.repeat_num, config.data_format,
                                          True)
    G2_out = tf.identity(G2_out, name='G2_output')
    tf.add_to_collection('outputs', G2_out)
    #tf.add_to_collection('trainable', G_vars)
    saver = tf.train.Saver()
    return saver



