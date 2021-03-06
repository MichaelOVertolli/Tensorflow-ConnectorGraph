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
    sizes = [min(2**(config.max_dist-i), config.hidden_num) for i in range(config.repeat_num)]
    scopes = ['D'+str(i) for i in range(config.repeat_num)]
    alphas = [tf.placeholder(tf.float32, (), name='alpha'+str(i)) for i in range(config.repeat_num-1)]
    for alpha in alphas:
        tf.add_to_collection('inputs', alpha)
    D_ins = []
    for i in range(config.repeat_num):
        x = tf.placeholder(tf.float32, [None, config.output_num,
                                        config.base_size*(2**i),
                                        config.base_size*(2**i)],
                           name=''.join(['input', str(i)]))
        tf.add_to_collection('inputs', x)
        D_ins.append(x)
    alphas.reverse()
    D_ins.reverse()
    sizes.reverse()
    len_scopes = len(scopes)
    for i, cur_scope in enumerate(scopes):
        scopes_ = scopes[::-1][len_scopes-(i+1):] #we build the discriminators in reverse order
        sizes_ = sizes[len_scopes-(i+1):]
        alphas_ = alphas[len_scopes-(i+1):] #empty on first iteration... one less alpha than len_scopes
        D_in = D_ins[len_scopes-(i+1)]
        D_out, variables = models.DiscriminatorNSkipCNN(D_in, sizes_, scopes_, alphas_, cur_scope,
                                                        config.hidden_num, config.data_format)
        D_out = tf.identity(D_out, name='output'+str(i))
        tf.add_to_collection('outputs', D_out)
        for variable in variables:
            tf.add_to_collection(cur_scope+'_tensors', variable)
    
    saver = tf.train.Saver()
    return saver



