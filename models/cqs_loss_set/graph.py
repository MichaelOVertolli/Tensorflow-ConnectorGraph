import numpy as np
import tensorflow as tf
from .. import models
from .. import friqa
from ..null_variable import make_null_variable


def build_graph(model_name, config):
    with tf.variable_scope(model_name):
        real = tf.placeholder(tf.float32, [None, None, None, None],
                              name='real_input')
        autoencoded = tf.placeholder(tf.float32, [None, None, None, None],
                                     name='autoencoded_input')
        l1 = tf.reduce_mean(tf.abs(autoencoded - real))
        gms, chrom = friqa.prep_and_call_qs(real, autoencoded)
        loss = (config.l1weight*l1 +
                config.gmsweight*gms +
                config.chromweight*chrom) / config.totalweight
        make_null_variable()
    tf.add_to_collection('inputs', real)
    tf.add_to_collection('inputs', autoencoded)
    tf.add_to_collection('outputs', loss)
    saver = tf.train.Saver()
    return saver



