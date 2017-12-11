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
REST = 'res_train'
CNCT = 'nvd_concat_0'
LOSS = 'res_loss_wass_0'
MIXD = 'nvd_mix_0'

#Inputs
INPT = '/input:0'
G_IN = '/gen_input:0'
D_IN = '/data_input:0'
M_IN = '/mix_input:0'

#Outputs
OUTP = '/output:0'
GOUT = '/g_loss:0'
DOUT = '/d_loss:0'
MOUT = '/mix_output:0'


#Alphas
ALPH = '/alpha{}:0'

#Variables
VARS = '/trainable_variables'

connections = [
    [GENR, CNCT, GENR+OUTP, CNCT+G_IN],
    [GENR, MIXD, GENR+OUTP, MIXD+G_IN],
    [MIXD, CNCT, MIXD+MOUT, CNCT+M_IN],
    [CNCT, DISC, CNCT+OUTP, DISC+INPT],
    [DISC, LOSS, DISC+OUTP, LOSS+INPT],
    [MIXD, LOSS, MIXD+MOUT, LOSS+M_IN],
]

inputs = [
    GENR+INPT,
    CNCT+D_IN,
    MIXD+D_IN,
]

outputs = [
    LOSS+GOUT,
    LOSS+DOUT,
]


def build_graph(config):
    #TODO: fix partial loading of saved variables from this model into partial models

    conngraph = ConnectorGraph(config)
    
    generator = init_subgraph(GENR, config.mdl_type)
    discriminator = init_subgraph(DISC, config.mdl_type)
    mixed_op = init_subgraph(MIXD, '')
    concat_op = init_subgraph(CNCT, '')
    loss_op = init_subgraph(LOSS, config.loss_type)

    conngraph.add_subgraph(generator)
    conngraph.add_subgraph(discriminator)
    conngraph.add_subgraph(mixed_op)
    conngraph.add_subgraph(concat_op)
    conngraph.add_subgraph(loss_op)

    if config.alphas:
        for i in range(config.repeat_num-1):
            inpts = [
                GENR+ALPH.format(i),
                DISC+ALPH.format(i),
            ]
            inputs.extend(inpts)
    
    conngraph.print_subgraphs()

    conngraph.quick_connect(connections)

    with tf.Session(graph=tf.Graph()) as sess:
        conngraph.connect_graph(inputs, outputs, sess)

        step = tf.Variable(0, dtype=tf.int32, name='step', trainable=False)
        
        with tf.variable_scope(REST):

            # incg = config.g_lr / 400.0
            # incd = config.d_lr / 400.0
            g_lr = tf.Variable(incg, name='g_lr')
            d_lr = tf.Variable(incd, name='d_lr')
            # g_lr_ramp = tf.assign(g_lr, tf.minimum(g_lr + incg, 1.0))
            # d_lr_ramp = tf.assign(d_lr, tf.minimum(d_lr + incd, 1.0))

            g_lr_update = tf.assign(g_lr, tf.maximum(g_lr * 1.0, config.lr_lower_boundary), name='g_lr_update')
            d_lr_update = tf.assign(d_lr, tf.maximum(d_lr * 1.0, config.lr_lower_boundary), name='d_lr_update')


            g_loss = tf.get_tensor_by_name(LOSS+GOUT)
            d_loss = tf.get_tensor_by_name(LOSS+DOUT)

            g_optimizer = tf.train.AdamOptimizer(g_lr)#, beta1=0.0)
            d_optimizer = tf.train.AdamOptimizer(d_lr)#, beta1=0.0)

            diter = tf.Variable(0, dtype=tf.int32, name='disc_iter', trainable=False)
            reset_diter = tf.assign(diter, 0)

            with tf.control_dependencies([reset_diter]):
                g_optim = g_optimizer.minimize(g_loss, global_step=step,
                                           var_list=tf.get_collection(GENR+VARS))

            d_vars = tf.get_collection(DISC+VARS)+tf.get_collection(LOSS+VARS)
            d_optim = d_optimizer.minimize(d_loss, var_list=d_vars)
            
            diter_check = lambda i: tf.less(i, config.disc_steps)
            with tf.control_dependencies([d_optim]):
                next_diter = lambda i: tf.assign(diter, diter+1)

            with tf.control_dependencies([g_optim]):
                out = tf.while_loop(diter_check, next_diter, [diter], parallel_iterations=1)

            # with tf.control_dependencies([d_optim, g_optim]):
            #     out = tf.reduce_mean(g_loss) - tf.reduce_mean(d_loss)
            #     out = tf.identity(out, name='output')

            # weight_updates = []
            # for var in tf.get_collection(VARS):
            #     if '/weights' in var.name:
            #         op = tf.assign(var, var/tf.sqrt(tf.reduce_mean(var*var)))
            #         weight_updates.append(op)

            # norm_weights = tf.group(*weight_updates, name='norm_weights')

            summaries = [
                tf.summary.image('G', denorm_img(sess.graph.get_tensor_by_name(GENR+OUTP), config.data_format)),
                tf.summary.image('D', denorm_img(sess.graph.get_tensor_by_name(CNCT+D_IN), config.data_format)),

                tf.summary.scalar('loss/g_loss', g_loss),
                tf.summary.scalar('loss/d_loss', d_loss),
            ]
            if config.alphas:
                for i in range(config.repeat_num-1):
                    summaries.append(tf.summary.scalar('misc/alpha_'+str(i), sess.graph.get_tensor_by_name(GENR+ALPH.format(i))))
                
            summary_op = tf.summary.merge(summaries)

            savers = {
                GENR: tf.train.Saver(sess.graph.get_collection('/'.join([GENR, 'variables']))),
                DISC: tf.train.Saver(sess.graph.get_collection('/'.join([DISC, 'variables'])))
            }

            conngraph.add_subgraph_savers(savers)

        conngraph.model_pos = 0
        conngraph.b_size = config.batch_size

        tf.add_to_collection('step', step)
        
        sess.graph.clear_collection('outputs')
        tf.add_to_collection('outputs_interim', d_loss)
        tf.add_to_collection('outputs_interim', g_loss)
        tf.add_to_collection('outputs_interim', summary_op)
        tf.add_to_collection('outputs', out)
        tf.add_to_collection('outputs', g_lr_ramp)
        tf.add_to_collection('outputs', d_lr_ramp)
        tf.add_to_collection('summary', summary_op)
        # tf.add_to_collection('norm_weights', norm_weights)
        tf.add_to_collection('outputs_lr', g_lr_update)
        tf.add_to_collection('outputs_lr', d_lr_update)

        def get_feed_dict(self, trainer):
            # hack to handle weight norming
            # _ = trainer.sess.run(trainer.c_graph.graph.get_collection('norm_weights')[0])

            step = trainer.sess.run(trainer.step)

            #model selection
            pos = int(step + config.alpha_update_step_size) / int(config.alpha_update_step_size*2)

            i_size = config.base_size*(2**pos)
            

            #feed_dict setup
            x = trainer.data_loader
            x = trainer.sess.run(x)

            x = norm_img(x) #running numpy version so don't have to modify graph
            z = np.random.uniform(-1, 1, size=(config.batch_size, trainer.z_num))
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
                feeds.append((MIXD+D_IN, img))
            else:
                feeds.append((CNCT+D_IN, x))
                feeds.append((MIXD+D_IN, x))
            feed_dict = dict(feeds)
            
            return feed_dict
        
        conngraph.attach_func(get_feed_dict)
        
        def send_outputs(self, trainer, step):
            if not hasattr(self, 'z_fixed'):
                self.z_fixed = np.random.uniform(-1, 1, size=(trainer.batch_size, trainer.z_num))

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
