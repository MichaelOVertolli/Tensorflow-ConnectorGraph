import numpy as np
import tensorflow as tf
from .. import models


def build_graph(model_name, config):
    GR_in = tf.placeholder(tf.float32, [None, config.output_num, config.size, config.size], name='input')
    GR_out, GR_vars = models.GeneratorRCNN(GR_in,
                                           config.output_num, config.z_num,
                                           config.repeat_num, config.hidden_num,
                                           config.data_format)
    GR_out = tf.identity(GR_out, name='output') 
    tf.add_to_collection('inputs', GR_in)
    tf.add_to_collection('outputs', GR_out)
    #tf.add_to_collection('trainable', G_vars)
    saver = tf.train.Saver()
    return saver



