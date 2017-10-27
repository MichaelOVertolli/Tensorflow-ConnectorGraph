from ..connectorgraph import ConnectorGraph
from ..data_loader import get_loader
from ..errors import FirstInitialization
import numpy as np
import os
from ..subgraph import BuiltSubGraph, SubGraph
import tensorflow as tf
from utils import save_image


#Models
GENB = 'cqs_generator_0'
GEN0 = 'frozen_cqs_generator_0'
GEN1 = 'frozen_cqs_generator_1'
GENR = 'cqs_reverse_0'
LOSS = 'cqs_loss_set_0'
CQST = 'cqs_train'

#Type variables
MDL_TYPE = 'z128_sz64'
LSS_TYPE = 'scaled_began_gmsm'

#Inputs
INPT = '/input:0'
F_IN = '/cqs_generator_0/input:0'
O_IN = '/orig_input:0'
A_IN = '/autoencoded_input:0'

#Outputs
OUTP = '/output:0'
FOUT = '/cqs_generator_0/output:0'

#Variables
VARS = '/trainable_variables'

connections = [
    #from_graph, to_graph, from_conn, to_conn
    [GEN0, GENR, GEN0+FOUT, GENR+INPT],
    [GENR, GEN1, GENR+OUTP, GEN1+F_IN],
    [GEN0, LOSS, GEN0+FOUT, LOSS+O_IN],
    [GEN1, LOSS, GEN1+FOUT, LOSS+A_IN],
]

inputs = [
    GEN0+F_IN,
]

outputs = [
    GEN0+FOUT,
    GEN1+FOUT,
    LOSS+OUTP,
]

GEN_LD = './logs/cqs_cg/cqs_generator_0/'


def build_graph(config):
    #TODO: fix partial loading of saved variables from this model into partial models
    with tf.Session(graph=tf.Graph()) as sess:
        generator_base = BuiltSubGraph(GENB, MDL_TYPE, sess, GEN_LD)
        generator_0 = generator_base.freeze(sess)
    
    generator_1 = generator_0.copy(GEN1)
    reverse_gen = init_subgraph(GENR, MDL_TYPE)
    loss_set = init_subgraph(LOSS, LSS_TYPE)

    conngraph = ConnectorGraph()
    conngraph.add_subgraph(generator_0)
    conngraph.add_subgraph(generator_1)
    conngraph.add_subgraph(reverse_gen)
    conngraph.add_subgraph(loss_set)

    conngraph.print_subgraphs()

    conngraph.quick_connect(connections)


    with tf.Session(graph=tf.Graph()) as sess:
        conngraph.connect_graph(inputs, outputs, sess)
        
        with tf.variable_scope('cqs_train'):
            loss = sess.graph.get_tensor_by_name(LOSS+OUTP)
            loss = tf.identity(loss, name='loss')

            lr = tf.Variable(config.lr, name='lr')

            lr_update = tf.assign(lr, tf.maximum(lr * 0.5, config.lr_lower_boundary), name='lr_update')

            optimizer = tf.train.AdamOptimizer(lr)

            optim = optimizer.minimize(loss, var_list=tf.get_collection(GENR+VARS))

            summary_op = tf.summary.merge([
                tf.summary.image('G', denorm_img(sess.graph.get_tensor_by_name(GEN0+FOUT), config.data_format)),
                tf.summary.image('GR', denorm_img(sess.graph.get_tensor_by_name(GEN1+FOUT), config.data_format)),
                tf.summary.scalar('loss/loss', loss),
                tf.summary.scalar('misc/g_lr', g_lr),
            ])

            #TODO: should reuse Savers from the original subgraph definitions
            savers = {
                GENR: tf.train.Saver(sess.graph.get_collection('/'.join([GENR, 'variables']))),
            }

            conngraph.add_subgraph_savers(savers)

        step = tf.Variable(0, name='step', trainable=False)
        tf.add_to_collection('step', step)
        
        sess.graph.clear_collection('outputs')
        tf.add_to_collection('outputs_interim', loss)
        tf.add_to_collection('outputs_interim', summary_op)
        tf.add_to_collection('outputs_lr', lr_update)
        tf.add_to_collection('summary', summary_op)

        def get_feed_dict(self, trainer):
            x = trainer.data_loader
            # x = norm_img(x)
            x = trainer.sess.run(x)
            x = norm_img(x) #running numpy version so don't have to modify graph
            z = np.random.uniform(-1, 1, size=(trainer.batch_size, trainer.z_num))

            feed_dict = {GEN0+F_IN: z}
            return feed_dict
        
        conngraph.attach_func(get_feed_dict)
        
        def send_outputs(self, trainer, step):
            if not hasattr(self, 'z_fixed'):
                self.z_fixed = np.random.uniform(-1, 1, size=(trainer.batch_size, trainer.z_num))
                self.x_fixed = trainer.get_image_from_loader()
                save_image(self.x_fixed, '{}/x_fixed.png'.format(trainer.log_dir))
                self.x_fixed = norm_img(self.x_fixed)

            #generate original
            x_gen = trainer.sess.run(trainer.sess.graph.get_tensor_by_name(GEN0+OUTP),
                                     {GEN0+F_IN: self.z_fixed})
            
            save_image(denorm_img_numpy(x_gen, trainer.data_format),
                       '{}/{}_G.png'.format(trainer.log_dir, step))
                
            #generate reversed
            x_genr = trainer.sess.run(trainer.sess.graph.get_tensor_by_name(GEN1+FOUT),
                                     {GEN1+F_IN: self.z_fixed})
            
            save_image(denorm_img_numpy(x_gen, trainer.data_format),
                       '{}/{}_GR.png'.format(trainer.log_dir, step))

        conngraph.attach_func(send_outputs)
        
    return conngraph


def denorm_img(norm, data_format):
    return tf.clip_by_value(to_nhwc((norm + 1)*127.5, data_format), 0, 255)


def denorm_img_numpy(norm, data_format):
    return np.clip(to_nhwc_numpy((norm + 1)*127.5, data_format), 0, 255)


def nchw_to_nhwc(x):
    return tf.transpose(x, [0, 2, 3, 1])


def to_nhwc(image, data_format):
    if data_format == 'NCHW':
        new_image = nchw_to_nhwc(image)
    else:
        new_image = image
    return new_image


def to_nhwc_numpy(image, data_format):
    if data_format == 'NCHW':
        new_image = image.transpose([0, 2, 3, 1])
    else:
        new_image = image
    return new_image


def to_nchw_numpy(image):
    if image.shape[3] in [1, 3]:
        new_image = image.transpose([0, 3, 1, 2])
    else:
        new_image = image
    return new_image


def norm_img(image, data_format=None):
    image = image/127.5 - 1.
    if data_format:
        image = to_nhwc_numpy(image, data_format)
    return image


def slerp(val, low, high):
    """Code from https://github.com/soumith/dcgan.torch/issues/14"""
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high


def init_subgraph(subgraph_name, type_):
    try:
        with tf.Session(graph=tf.Graph()) as sess:
            subgraph = BuiltSubGraph(subgraph_name, type_, sess)
    except FirstInitialization:
        with tf.Session(graph=tf.Graph()) as sess:
            subgraph = BuiltSubGraph(subgraph_name, type_, sess)
    return subgraph
