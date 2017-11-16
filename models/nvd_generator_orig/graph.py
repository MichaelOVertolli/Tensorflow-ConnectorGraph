import numpy as np
import tensorflow as tf
from .. import models


def build_graph(model_name, config):
    # alphas = [tf.Variable(0.1, trainable=False, name='alpha'+str(i), dtype=tf.float32)
    #           for i in range(config.repeat_num-1)]
    alphas = [tf.placeholder(tf.float32, (), name='alpha'+str(i)) for i in range(config.repeat_num-1)]
    G0_in = tf.placeholder(tf.float32, [None, config.z_num], name='input0')
    G0_outs, G0_vars = models.GeneratorNSkipCNN(G0_in,
                                                config.hidden_num, config.output_num,
                                                config.repeat_num, alphas,
                                                config.data_format, False)
    G_ins, G_outs, G_vars = [G0_in], [G0_outs], [G0_vars]
    for i in range(1, config.repeat_num):
        G_ins.append(tf.placeholder(tf.float32, [None, config.z_num], name='input'+str(i)))
        temp_outs, temp_vars = models.GeneratorNSkipCNN(G_ins[i],
                                                        config.hidden_num, config.output_num,
                                                        config.repeat_num, alphas,
                                                        config.data_format, True)
    for alpha in alphas:
        tf.add_to_collection('inputs', alpha)
    for G_in in G_ins:
        tf.add_to_collection('inputs', G_in)
    for j, G_out in enumerate(G_outs):
        for i in range(len(G_out)):
            temp = tf.identity(G_outs[i], name='_'.join(['output', str(j), str(i)]))
            tf.add_to_collection('outputs', temp)
    saver = tf.train.Saver()
    return saver



