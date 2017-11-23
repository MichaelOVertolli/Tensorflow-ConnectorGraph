###############################################################################
#Copyright (C) 2017  Michael O. Vertolli michaelvertolli@gmail.com
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program.  If not, see http://www.gnu.org/licenses/
###############################################################################

from ..connectorgraph import ConnectorGraph
from ..errors import FirstInitialization
import numpy as np
import os
from ..subgraph import BuiltSubGraph, SubGraph
import tensorflow as tf
from ..model_utils import *


#Models
GENR = 'cqs_generator_loop_0'
DISC = 'cqs_discriminator_0'
LSSG = 'cqs_loss_set_0'
LSSD = 'cqs_loss_set_1'
LSSR = 'cqs_loss_set_3'
CNCT = 'cqs_concat_0'
SPLT = 'cqs_split_0'
CQST = 'cqs_train'

#Inputs
INPT = '/input:0'
G1IN = '/G1_input:0'
GRIN = '/GR_input:0'
G_IN = '/gen_input:0'
D_IN = '/data_input:0'
O_IN = '/orig_input:0'
A_IN = '/autoencoded_input:0'

#Outputs
OUTP = '/output:0'
G1OT = '/G1_output:0'
G2OT = '/G2_output:0'
GROT = '/GR_output:0'
GOUT = '/gen_output:0'
DOUT = '/data_output:0'

#Variables
VARS = '/trainable_variables'

#Tensors
G_TN = '/G_tensors'
GRTN = '/GR_tensors'

connections = [
    #from_graph, to_graph, from_conn, to_conn
    [GENR, CNCT, GENR+G1OT, CNCT+G_IN],
    [CNCT, DISC, CNCT+OUTP, DISC+INPT],
    [DISC, SPLT, DISC+OUTP, SPLT+INPT],
    [GENR, LSSG, GENR+G1OT, LSSG+O_IN],
    [SPLT, LSSG, SPLT+GOUT, LSSG+A_IN],
    [SPLT, LSSD, SPLT+DOUT, LSSD+A_IN],
    [GENR, LSSR, GENR+G1OT, LSSR+O_IN],
    [GENR, LSSR, GENR+G2OT, LSSR+A_IN],
]

inputs = [
    GENR+G1IN,
    GENR+GRIN,
    CNCT+D_IN,
    LSSD+O_IN, #same as CNCT+D_IN
]

outputs = [
    GENR+G1OT,
    GENR+G2OT,
    GENR+GROT,
    LSSG+OUTP,
    LSSD+OUTP,
    LSSR+OUTP,
]


def build_graph(config):
    #TODO: fix partial loading of saved variables from this model into partial models

    generator = init_subgraph(GENR, config.mdl_type)
    discriminator = init_subgraph(DISC, config.mdl_type)
    disc_loss_set = init_subgraph(LSSD, config.lss_type)
    gen_loss_set = init_subgraph(LSSG, config.lss_type)
    rev_loss_set = init_subgraph(LSSR, config.lss_type)
    concat_op = init_subgraph(CNCT, '')
    split_op = init_subgraph(SPLT, '')
    
    
    conngraph = ConnectorGraph(config)
    conngraph.add_subgraph(generator)
    conngraph.add_subgraph(discriminator)
    conngraph.add_subgraph(disc_loss_set)
    conngraph.add_subgraph(gen_loss_set)
    conngraph.add_subgraph(rev_loss_set)
    conngraph.add_subgraph(concat_op)
    conngraph.add_subgraph(split_op)

    conngraph.print_subgraphs()

    conngraph.quick_connect(connections)


    with tf.Session(graph=tf.Graph()) as sess:
        conngraph.connect_graph(inputs, outputs, sess)
        
        with tf.variable_scope('cqs_train'):
            k_t = tf.Variable(0., trainable=False, name='k_t')

            d_loss = sess.graph.get_tensor_by_name(LSSD+OUTP)
            g_loss = sess.graph.get_tensor_by_name(LSSG+OUTP)
            gr_loss = sess.graph.get_tensor_by_name(LSSR+OUTP)
            g_loss = tf.identity(g_loss, name='g_loss')
            gr_loss = tf.identity(gr_loss, name='gr_loss')

            d_out = d_loss - k_t * g_loss
            d_out = tf.identity(d_out, name='d_loss')

            d_lr = tf.Variable(config.d_lr, name='d_lr')
            g_lr = tf.Variable(config.g_lr, name='g_lr')
            gr_lr = tf.Variable(config.gr_lr, name='gr_lr')
            
            d_lr_update = tf.assign(d_lr, tf.maximum(d_lr * 0.5, config.lr_lower_boundary), name='d_lr_update')
            g_lr_update = tf.assign(g_lr, tf.maximum(g_lr * 0.5, config.lr_lower_boundary), name='g_lr_update')
            gr_lr_update = tf.assign(gr_lr, tf.maximum(gr_lr * 0.5, config.lr_lower_boundary), name='gr_lr_update')

            d_optimizer = tf.train.AdamOptimizer(d_lr)
            g_optimizer = tf.train.AdamOptimizer(g_lr)
            gr_optimizer = tf.train.AdamOptimizer(gr_lr)

            d_optim = d_optimizer.minimize(d_out, var_list=tf.get_collection(DISC+VARS))
            g_optim = g_optimizer.minimize(g_loss, var_list=conngraph.get_all_variables(
                conngraph.tensors_to_names(
                    tf.get_collection(GENR+G_TN))))
            gr_optim = gr_optimizer.minimize(gr_loss, var_list=conngraph.get_all_variables(
                conngraph.tensors_to_names(
                    tf.get_collection(GENR+GRTN))))

            balance = config.gamma * d_loss - g_loss
            measure = d_loss + tf.abs(balance)
            measure = tf.identity(measure, name='measure')

            with tf.control_dependencies([d_optim, g_optim, gr_optim]):
                k_update = tf.assign(k_t, tf.clip_by_value(k_t + config.lambda_k * balance, 0, 1))
                k_update = tf.identity(k_update, name='k_update')

            summary_op = tf.summary.merge([
                tf.summary.image('G', denorm_img(sess.graph.get_tensor_by_name(GENR+G1OT), config.data_format)),
                tf.summary.image('GR', denorm_img(sess.graph.get_tensor_by_name(GENR+G2OT), config.data_format)),
                tf.summary.image('AE_G', denorm_img(sess.graph.get_tensor_by_name(SPLT+GOUT), config.data_format)),
                tf.summary.image('AE_D', denorm_img(sess.graph.get_tensor_by_name(SPLT+DOUT), config.data_format)),

                tf.summary.scalar('loss/d_loss', d_out),
                tf.summary.scalar('loss/g_loss', g_loss),
                tf.summary.scalar('loss/gr_loss', gr_loss),

                tf.summary.scalar('misc/measure', measure),
                tf.summary.scalar('misc/k_t', k_t),
                tf.summary.scalar('misc/d_lr', d_lr),
                tf.summary.scalar('misc/g_lr', g_lr),
                tf.summary.scalar('misc/gr_lr', gr_lr),
                tf.summary.scalar('misc/balance', balance),
            ])

            #TODO: should reuse Savers from the original subgraph definitions
            savers = {
                GENR: tf.train.Saver(sess.graph.get_collection('/'.join([GENR, 'variables']))),
                DISC: tf.train.Saver(sess.graph.get_collection('/'.join([DISC, 'variables'])))
            }

            conngraph.add_subgraph_savers(savers)

        step = tf.Variable(0, name='step', trainable=False)
        tf.add_to_collection('step', step)
        
        sess.graph.clear_collection('outputs')
        tf.add_to_collection('outputs_interim', d_out)
        tf.add_to_collection('outputs_interim', g_loss)
        tf.add_to_collection('outputs_interim', gr_loss)
        tf.add_to_collection('outputs_interim', k_t)
        tf.add_to_collection('outputs_interim', summary_op)
        tf.add_to_collection('outputs', k_update)
        tf.add_to_collection('outputs', measure)
        tf.add_to_collection('outputs_lr', g_lr_update)
        tf.add_to_collection('outputs_lr', d_lr_update)
        tf.add_to_collection('summary', summary_op)

        def get_feed_dict(self, trainer):
            x = trainer.data_loader
            # x = norm_img(x)
            x = trainer.sess.run(x)
            x = norm_img(x) #running numpy version so don't have to modify graph
            z = np.random.uniform(-1, 1, size=(trainer.batch_size, trainer.z_num))

            feed_dict = {GENR+G1IN: z, 
                         CNCT+D_IN: x,
                         LSSD+O_IN: x}
            return feed_dict
        
        conngraph.attach_func(get_feed_dict)
        
        def send_outputs(self, trainer, step):
            if not hasattr(self, 'z_fixed'):
                self.z_fixed = np.random.uniform(-1, 1, size=(trainer.batch_size, trainer.z_num))
                self.x_fixed = trainer.get_image_from_loader()
                save_image(self.x_fixed, os.path.join(trainer.log_dir, 'x_fixed.png'))
                self.x_fixed = norm_img(self.x_fixed)

            #generate
            x_gen = trainer.sess.run(trainer.sess.graph.get_tensor_by_name(GENR+G1OT),
                                     {GENR+G1IN: self.z_fixed})
            
            save_image(denorm_img_numpy(x_gen, trainer.data_format),
                       os.path.join(trainer.log_dir, '{}_G.png'.format(step)))

            x_gen2 = trainer.sess.run(trainer.sess.graph.get_tensor_by_name(GENR+G2OT),
                                      {GENR+G1IN: self.z_fixed})
            
            save_image(denorm_img_numpy(x_gen2, trainer.data_format),
                       os.path.join(trainer.log_dir, '{}_GR.png'.format(step)))

            #autoencode
            for k, img in (('real', self.x_fixed), ('gen', x_gen)):
                if img is None:
                    continue
                if img.shape[3] in [1, 3]:
                    img = img.transpose([0, 3, 1, 2])
                x = trainer.sess.run(trainer.sess.graph.get_tensor_by_name(SPLT+DOUT),
                                     {GENR+G1IN: self.z_fixed,
                                      CNCT+D_IN: img})
                save_image(denorm_img_numpy(x, trainer.data_format),
                           os.path.join(trainer.log_dir, '{}_D_{}.png'.format(step, k)))


            #interpolate
            z_flex = np.random.uniform(-1, 1, size=(trainer.batch_size, trainer.z_num))
            generated = []
            for _, ratio in enumerate(np.linspace(0, 1, 10)):
                z = np.stack([slerp(ratio, r1, r2) for r1, r2 in zip(self.z_fixed, z_flex)])
                #generate
                z_decode = trainer.sess.run(trainer.sess.graph.get_tensor_by_name(GENR+G1OT),
                                            {GENR+G1IN: z})
                generated.append(denorm_img_numpy(z_decode, trainer.data_format))

            generated = np.stack(generated).transpose([1, 0, 2, 3, 4])

            all_img_num = np.prod(generated.shape[:2])
            batch_generated = np.reshape(generated, [all_img_num] + list(generated.shape[2:]))
            save_image(batch_generated, os.path.join(trainer.log_dir, '{}_interp_G.png'.format(step)), nrow=10)
            

        conngraph.attach_func(send_outputs)
        
    return conngraph
