import numpy as np
import tensorflow as tf
from .. import models


def build_graph(model_name, config):
    graph = tf.Graph()
    with graph.as_default():
        with tf.variable_scope(model_name):
            D_in = tf.placeholder(tf.float32, [None, config.output_num, config.size, config.size])
            D_out, D_z, D_vars = models.DiscriminatorCNN(D_in,
                                                         config.output_num, config.z_num,
                                                         config.repeat_num, config.hidden_num,
                                                         config.data_format)
        tf.add_to_collection('inputs', D_in)
        tf.add_to_collection('outputs', D_out)
        #tf.add_to_collection('trainable', G_vars)
        saver = tf.train.Saver()
    return graph, saver



