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

import tensorflow as tf
from ..null_variable import *


def build_graph(model_name, config):
    inpt = tf.placeholder(tf.float32, [None, None, None, None], name='input')
    data, gen = tf.split(inpt, 2)
    data = tf.identity(data, name='data_output')
    gen = tf.identity(gen, name='gen_output')
    tf.add_to_collection('inputs', inpt)
    tf.add_to_collection('outputs', data)
    tf.add_to_collection('outputs', gen)
    make_null_variable()
    saver = tf.train.Saver()
    return saver
