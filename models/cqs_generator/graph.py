import numpy as np
import tensorflow as tf
from .. import models


def build_graph(model_name, config):
    G_in = tf.placeholder(tf.float32, [config.batch, config.z_num], name='input')
    G_out, G_vars = models.GeneratorCNN(G_in,
                                        config.hidden_num, config.output_num,
                                        config.repeat_num, config.data_format,
                                        config.reuse)
    G_out = tf.identity(G_out, name='output')
    tf.add_to_collection('inputs', G_in)
    tf.add_to_collection('outputs', G_out)
    #tf.add_to_collection('trainable', G_vars)
    saver = tf.train.Saver()
    return saver



