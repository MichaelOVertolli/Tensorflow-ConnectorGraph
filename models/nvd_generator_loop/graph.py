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
    for v in G_vars:
        tf.add_to_collection('G_tensors', v)

    GR_in = tf.identity(G_outs[-1], name='GR_input')
    GR_out, GR_vars = models.GeneratorRCNN(GR_in,
                                           config.output_num, config.z_num,
                                           config.repeat_num, config.hidden_num,
                                           config.data_format)
    GR_out = tf.identity(GR_out, name='GR_output')
    tf.add_to_collection('inputs', GR_in)
    tf.add_to_collection('outputs', GR_out)
    for v in GR_vars:
        tf.add_to_collection('GR_tensors', v)
    G2_outs, G2_vars = models.GeneratorSkipCNN(GR_out,
                                               config.hidden_num, config.output_num,
                                               config.repeat_num, alphas,
                                               config.data_format, True)
    G2_out = tf.identity(G2_outs[-1], name='G2_output')
    tf.add_to_collection('outputs', G2_out)
    
    saver = tf.train.Saver()
    return saver



