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
    alphas = [tf.placeholder(tf.float32, (), name='alpha'+str(i)) for i in range(config.repeat_num-1)]
    G_in = tf.placeholder(tf.float32, [None, config.z_num], name='input')
    G0_outs, G0_vars = models.GeneratorNSkipCNN(G_in,
                                                config.hidden_num, config.output_num,
                                                config.repeat_num, alphas,
                                                config.data_format, False)
    G_outs, G_vars = [G0_outs], [G0_vars]
    for i in range(1, config.repeat_num-1):
        temp_outs, temp_vars = models.GeneratorNSkipCNN(G_in,
                                                        config.hidden_num, config.output_num,
                                                        i, alphas, config.data_format, True)
        G_outs.append(temp_outs)
        G_vars.append(temp_vars)
        
    for alpha in alphas:
        tf.add_to_collection('inputs', alpha)
    tf.add_to_collection('inputs', G_in)
    for j, G_out in enumerate(G_outs):
        for i in range(len(G_out)):
            temp = tf.identity(G_out[i], name='_'.join(['output', str(j), str(i)]))
            tf.add_to_collection('outputs', temp)
    saver = tf.train.Saver()
    return saver



