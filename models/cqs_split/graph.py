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
