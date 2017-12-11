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
from skimage.measure import block_reduce
from scipy.ndimage import zoom
from ..subgraph import BuiltSubGraph, SubGraph
import tensorflow as tf
from ..model_utils import *
import time

#Models
GENR = 'res_generator_0'
DISC = 'res_discriminator_0'
DGEN = 'res_generator_1'
REST = 'res_train'
CNCT = 'cqs_concat_0'
SPLT = 'cqs_split_0'
LSSG = 'cqs_loss_set_0'
LSSD = 'cqs_loss_set_1'

#Inputs
INPT = '/input:0'
G_IN = '/gen_input:0'
D_IN = '/data_input:0'
M_IN = '/mix_input:0'
O_IN = '/orig_input:0'
A_IN = '/autoencoded_input:0'

#Outputs
OUTP = '/output:0'
GOUT = '/gen_output:0'
DOUT = '/data_output:0'
MOUT = '/mix_output:0'

#Alphas
ALPH = '/alpha{}:0'

#Variables
VARS = '/trainable_variables'
VARIABLES = '/variables'

connections = [
    [GENR, CNCT, GENR+OUTP, CNCT+G_IN],
    [CNCT, DISC, CNCT+OUTP, DISC+INPT],
    [DISC, DGEN, DISC+OUTP, DGEN+INPT],
    [DGEN, SPLT, DGEN+OUTP, SPLT+INPT],
    [GENR, LSSG, GENR+OUTP, LSSG+O_IN],
    [SPLT, LSSG, SPLT+GOUT, LSSG+A_IN],
    [SPLT, LSSD, SPLT+DOUT, LSSD+A_IN],
]

inputs = [
    GENR+INPT,
    CNCT+D_IN,
    LSSD+O_IN,
]

outputs = [
    LSSG+OUTP,
    LSSD+OUTP,
]


def build_graph(config):
    #TODO: fix partial loading of saved variables from this model into partial models

    conngraph = ConnectorGraph(config)
    
    generator = init_subgraph(GENR, config.mdl_type)
    discriminator = init_subgraph(DISC, config.mdl_type)
    discriminator_gen = init_subgraph(DGEN, config.mdl_type)
    concat_op = init_subgraph(CNCT, '')
    split_op = init_subgraph(SPLT, '')
    disc_loss_set = init_subgraph(LSSD, config.lss_type)
    gen_loss_set = init_subgraph(LSSG, config.lss_type)

    conngraph.add_subgraph(generator)
    conngraph.add_subgraph(discriminator)
    conngraph.add_subgraph(discriminator_gen)
    conngraph.add_subgraph(concat_op)
    conngraph.add_subgraph(split_op)
    conngraph.add_subgraph(gen_loss_set)
    conngraph.add_subgraph(disc_loss_set)        

    if config.alphas:
        for i in range(config.repeat_num-1):
            inputs.extend([
                GENR+ALPH.format(i),
                DISC+ALPH.format(i),
                DGEN+ALPH.format(i),
            ])
    
    conngraph.print_subgraphs()

    conngraph.quick_connect(connections)

    with tf.Session(graph=tf.Graph()) as sess:
        conngraph.connect_graph(inputs, outputs, sess)

        step = tf.Variable(0, dtype=tf.int32, name='step', trainable=False)
        
        with tf.variable_scope(REST):

            k_t = tf.Variable(0., trainable=False, name='k_t')

            d_loss = sess.graph.get_tensor_by_name(LSSD+OUTP)
            g_loss = sess.graph.get_tensor_by_name(LSSG+OUTP)
            g_loss = tf.identity(g_loss, name='g_loss')

            d_out = d_loss - k_t * g_loss
            d_out = tf.identity(d_out, name='d_loss')

            g_lr = tf.Variable(config.g_lr, name='g_lr')
            d_lr = tf.Variable(config.d_lr, name='d_lr')

            g_lr_update = tf.assign(g_lr, tf.maximum(g_lr * 0.5, config.lr_lower_boundary), name='g_lr_update')
            d_lr_update = tf.assign(d_lr, tf.maximum(d_lr * 0.5, config.lr_lower_boundary), name='d_lr_update')

            g_optimizer = tf.train.AdamOptimizer(g_lr)
            d_optimizer = tf.train.AdamOptimizer(d_lr)

            # TODO: add incremental variable training as separate mod
            g_optim = g_optimizer.minimize(g_loss, var_list=tf.get_collection(GENR+VARS))
            D_vars = tf.get_collection(DISC+VARS)+tf.get_collection(DGEN+VARS)
            d_optim = d_optimizer.minimize(d_out, var_list=D_vars)

            balance = config.gamma * d_loss - g_loss
            measure = d_loss + tf.abs(balance)
            measure = tf.identity(measure, name='measure')

            with tf.control_dependencies([d_optim, g_optim]):
                k_update = tf.assign(k_t, tf.clip_by_value(k_t + config.lambda_k * balance, 0, 1))
                k_update = tf.identity(k_update, name='k_update')

            summary_op = tf.summary.merge([
                tf.summary.image('G', denorm_img(sess.graph.get_tensor_by_name(GENR+OUTP), config.data_format)),
                tf.summary.image('AE_G', denorm_img(sess.graph.get_tensor_by_name(SPLT+GOUT), config.data_format)),
                tf.summary.image('AE_D', denorm_img(sess.graph.get_tensor_by_name(SPLT+DOUT), config.data_format)),

                tf.summary.scalar('loss/d_loss', d_out),
                tf.summary.scalar('loss/g_loss', g_loss),

                tf.summary.scalar('misc/measure', measure),
                tf.summary.scalar('misc/k_t', k_t),
                tf.summary.scalar('misc/g_lr', g_lr),
                tf.summary.scalar('misc/d_lr', d_lr),
                tf.summary.scalar('misc/balance', balance),
            ])

            savers = {
                GENR: tf.train.Saver(sess.graph.get_collection(GENR+VARIABLES)),
                DISC: tf.train.Saver(sess.graph.get_collection(DISC+VARIABLES)),
                DGEN: tf.train.Saver(sess.graph.get_collection(DGEN+VARIABLES)),
            }

            conngraph.add_subgraph_savers(savers)

        conngraph.model_pos = 0
        conngraph.b_size = config.batch_size

        tf.add_to_collection('step', step)
        
        sess.graph.clear_collection('outputs')
        sess.graph.clear_collection('outputs')
        tf.add_to_collection('outputs_interim', d_out)
        tf.add_to_collection('outputs_interim', g_loss)
        tf.add_to_collection('outputs_interim', k_t)
        tf.add_to_collection('outputs_interim', summary_op)
        tf.add_to_collection('outputs', k_update)
        tf.add_to_collection('outputs', measure)
        tf.add_to_collection('outputs_lr', g_lr_update)
        tf.add_to_collection('outputs_lr', d_lr_update)
        tf.add_to_collection('summary', summary_op)

        def get_feed_dict(self, trainer):

            step = trainer.sess.run(trainer.step)

            #model selection
            pos = int(step + config.alpha_update_step_size) / int(config.alpha_update_step_size*2)

            i_size = config.base_size*(2**pos)

            #feed_dict setup
            x = trainer.data_loader
            x = trainer.sess.run(x)
            x = norm_img(x) #running numpy version so don't have to modify graph
            z = np.random.uniform(-1, 1, size=(trainer.batch_size, trainer.z_num))

            feeds = [
                (GENR+INPT, z), 
            ]

            if trainer.c_graph.config.alphas:
                self.alphas_feed = []
                alphas = []
                for i in range(config.repeat_num-1):
                    val = step - config.alpha_update_steps*(2*(i+1) - 1)
                    if val < 0:
                        num = 0.0
                    else:
                        num = np.min([(val // config.alpha_update_step_size)*0.1, 1.0])
                    self.alphas_feed.append((GENR+ALPH.format(i), num))
                    self.alphas_feed.append((DISC+ALPH.format(i), num))
                    self.alphas_feed.append((DGEN+ALPH.format(i), num))
                    alphas.append(num)


                feeds.extend(self.alphas_feed)

            
                imgs = []
                block_size = config.img_size / i_size
                img = block_reduce(x, (1, 1, block_size, block_size), np.mean)
                img = zoom(img, [1, 1, block_size, block_size], mode='nearest')


                if pos != 0:
                    alpha = alphas[pos-1]
                    if alpha < 1.0: #then we are in a transition stage
                        block_size = config.img_size / (i_size/2) #get previous image size
                        img2 = block_reduce(x, (1, 1, block_size, block_size), np.mean)
                        img2 = zoom(img, [1, 1, block_size, block_size], mode='nearest')
                        img = img2*(1-alpha) + img*alpha #and bake it into the input image

                feeds.append((CNCT+D_IN, img))
                feeds.append((LSSD+O_IN, img))

            else:
                feeds.extend([
                    (CNCT+D_IN, x),
                    (LSSD+O_IN, x),
                ])
            feed_dict = dict(feeds)
            
            return feed_dict
        
        conngraph.attach_func(get_feed_dict)
        
        def send_outputs(self, trainer, step):
            if not hasattr(self, 'z_fixed'):
                self.z_fixed = np.random.uniform(-1, 1, size=(trainer.batch_size, trainer.z_num))
                self.x_fixed = trainer.get_image_from_loader()
                save_image(self.x_fixed, os.path.join(trainer.log_dir, 'x_fixed.png'))
                self.x_fixed = norm_img(self.x_fixed)

            pos = trainer.c_graph.model_pos
            alphas = trainer.c_graph.config.alphas

            #generate
            z_fixed = self.z_fixed
            feeds = [(GENR+INPT, z_fixed)]
            if alphas:
                feeds.extend(self.alphas_feed)
            feeds = dict(feeds)
            x_gen = trainer.sess.run(trainer.sess.graph.get_tensor_by_name(GENR+OUTP), feeds)

            save_image(denorm_img_numpy(x_gen, trainer.data_format),
                       os.path.join(trainer.log_dir, '{}_G.png'.format(step)))

            #autoencode
            for k, img in (('real', self.x_fixed), ('gen', x_gen)):
                if img is None:
                    continue
                if img.shape[3] in [1, 3]:
                    img = img.transpose([0, 3, 1, 2])
                afeeds = [
                    (GENR+INPT, z_fixed),
                    (CNCT+D_IN, img),
                ]
                if alphas:
                    afeeds.extend(self.alphas_feed)
                afeeds = dict(afeeds)
                x = trainer.sess.run(trainer.sess.graph.get_tensor_by_name(SPLT+DOUT),
                                     afeeds)
                save_image(denorm_img_numpy(x, trainer.data_format),
                           os.path.join(trainer.log_dir, '{}_D_{}.png'.format(step, k)))

            #interpolate
            z_flex = np.random.uniform(-1, 1, size=(trainer.batch_size, trainer.z_num))
            generated = []
            for _, ratio in enumerate(np.linspace(0, 1, 10)):
                z = np.stack([slerp(ratio, r1, r2) for r1, r2 in zip(z_fixed, z_flex)])
                #generate
                feeds = [(GENR+INPT, z)]
                if alphas:
                    feeds.extend(self.alphas_feed)
                feeds = dict(feeds)
                z_decode = trainer.sess.run(trainer.sess.graph.get_tensor_by_name(GENR+OUTP),
                                            feeds)
                generated.append(denorm_img_numpy(z_decode, trainer.data_format))

            generated = np.stack(generated).transpose([1, 0, 2, 3, 4])

            all_img_num = np.prod(generated.shape[:2])
            batch_generated = np.reshape(generated, [all_img_num] + list(generated.shape[2:]))
            save_image(batch_generated, os.path.join(trainer.log_dir, '{}_interp_G.png'.format(step)), nrow=10)

        conngraph.attach_func(send_outputs)
        
    return conngraph
