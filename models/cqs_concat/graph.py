import tensorflow as tf
from ..null_variable import *


def build_graph(model_name, config):
    data = tf.placeholder(tf.float32, [None, None, None, None], name='data_input')
    gen = tf.placeholder(tf.float32, [None, None, None, None], name='gen_input')
    concat = tf.concat([data, gen], 0, name='output')
    tf.add_to_collection('inputs', data)
    tf.add_to_collection('inputs', gen)
    tf.add_to_collection('outputs', concat)
    make_null_variable()
    saver = tf.train.Saver()
    return saver
