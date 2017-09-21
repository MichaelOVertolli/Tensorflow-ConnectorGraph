import numpy as np
import tensorflow as tf
from .. import models


def build_graph(model_name, config):
    D_in = tf.placeholder(tf.float32, [None, config.output_num, config.size, config.size], name='input')
    D_out, D_z, D_vars = models.DiscriminatorCNN(D_in,
                                                 config.output_num, config.z_num,
                                                 config.repeat_num, config.hidden_num,
                                                 config.data_format)
    D_out = tf.identity(D_out, name='output') 
    tf.add_to_collection('inputs', D_in)
    tf.add_to_collection('outputs', D_out)
    #tf.add_to_collection('trainable', G_vars)
    saver = tf.train.Saver()
    return saver



