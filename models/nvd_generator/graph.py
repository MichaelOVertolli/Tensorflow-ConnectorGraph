import numpy as np
import tensorflow as tf
from .. import models


def build_graph(model_name, config):
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



