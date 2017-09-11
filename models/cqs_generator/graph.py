import numpy as np
import tensorflow as tf
from .. import models


def build_graph(config):
    graph = tf.Graph()
    with graph.as_default():
        G_in = tf.placeholder(tf.float32, [None, config.z_num])
        G_out, G_vars = models.GeneratorCNN(G_in,
                                            config.hidden_num, config.output_num,
                                            config.repeat_num, config.data_format,
                                            config.reuse)
        tf.add_to_collection('inputs', G_in)
        tf.add_to_collection('outputs', G_out)
        #tf.add_to_collection('trainable', G_vars)
        saver = tf.train.Saver()
    return graph, saver



