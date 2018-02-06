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
from ..graphs.converter import build_variables
import numpy as np
import os
from skimage.measure import block_reduce
from scipy.ndimage import zoom
from ..subgraph import BuiltSubGraph, SubGraph
import tensorflow as tf
from ..model_utils import *
import time


def build_train_ops(conngraph, inputs, outputs,
                    train_scope, loss_tensors, train_sets,
                    img_pairs, saver_pairs, alpha_tensor, **keys):
    config = conngraph.config
    with tf.Session(graph=tf.Graph()) as sess:
        conngraph.connect_graph(inputs, outputs, sess)

        variables = build_variables(conngraph, sess, train_sets) # build_variables needs to be adjusted
        
        step = tf.Variable(0, dtype=tf.int32, name='step', trainable=False)
        
        with tf.variable_scope(train_scope):

            k_t = tf.Variable(0., trainable=False, name='k_t')

            u_losses = [sess.graph.get_tensor_by_name(loss_tensor) for loss_tensor in loss_tensors['U']]
            g_losses = [sess.graph.get_tensor_by_name(loss_tensor) for loss_tensor in loss_tensors['G']]
            g_losses = [g_losses[i]+(conngraph.config.lambda_u*u_losses[i])
                        for i in range(len(g_losses))] # add diversity loss to gen loss
            r_losses = [sess.graph.get_tensor_by_name(loss_tensor) for loss_tensor in loss_tensors['R']]
            g_losses = [tf.identity(g_loss, name='g_loss{}'.format(i)) for i, g_loss in enumerate(g_losses)]
            r_losses = [tf.identity(r_loss, name='r_loss{}'.format(i)) for i, r_loss in enumerate(r_losses)]

            d_loss = sess.graph.get_tensor_by_name(loss_tensors['D'])
            d_out = d_loss - k_t * g_loss
            d_out = tf.identity(d_out, name='d_loss')

            g_lr = tf.Variable(config.g_lr, name='g_lr')
            r_lr = tf.Variable(config.r_lr, name='r_lr')
            d_lr = tf.Variable(config.d_lr, name='d_lr')

            g_lr_update = tf.assign(g_lr, tf.maximum(g_lr * 0.5, config.lr_lower_boundary), name='g_lr_update')
            r_lr_update = tf.assign(r_lr, tf.maximum(r_lr * 0.5, config.lr_lower_boundary), name='r_lr_update')
            d_lr_update = tf.assign(d_lr, tf.maximum(d_lr * 0.5, config.lr_lower_boundary), name='d_lr_update')

            g_optimizers = [tf.train.AdamOptimizer(g_lr) for _ in g_losses]
            r_optimizers = [tf.train.AdamOptimizer(r_lr) for _ in r_losses]
            d_optimizer = tf.train.AdamOptimizer(d_lr)

            # TODO: add incremental variable training as separate mod
            g_optims = [g_optimizer.minimize(g_losses[i], var_list=variables['G'][i]) for i, g_optimizer in enumerate(g_optimizers)]
            r_optims = [r_optimizer.minimize(r_losses[i], var_list=variables['R'][i]) for i, r_optimizer in enumerate(r_optimizers)]
            
            d_optim = d_optimizer.minimize(d_out, global_step=step, var_list=variables['D'])

            g_optims = tf.group(g_optims)
            r_optims = tf.group(r_optims)

            balance = config.gamma * d_loss - tf.reduce_mean(g_losses)
            measure = d_loss + tf.abs(balance)
            measure = tf.identity(measure, name='measure')

            with tf.control_dependencies([d_optim, g_optims, r_optims]):
                k_update = tf.assign(k_t, tf.clip_by_value(k_t + config.lambda_k * balance, 0, 1))
                k_update = tf.identity(k_update, name='k_update')

            summary_set = []

            for name, tensor in img_pairs:
                summary_set.append(
                    tf.summary.image(name,
                                     denorm_img(
                                         sess.graph.get_tensor_by_name(tensor),
                                         config.data_format))
                )

            for i in range(conngraph.config.repeat_num-1):
                summary_set.append(tf.summary.scalar('misc/alpha_'+str(i), sess.graph.get_tensor_by_name(alpha_tensor.format(i))))

            for i in range(g_losses):
                summary_set.append(tf.summary.scalar('loss/g_loss{}'.format(i), g_losses[i]))
                tf.add_to_collection('outputs_interim', g_losses[i])

            for i in range(r_losses):
                summary_set.append(tf.summary.scalar('loss/r_loss{}'.format(i), r_losses[i]))
                tf.add_to_collection('outputs_interim', r_losses[i])

            summary_set.extend([
                tf.summary.scalar('loss/d_loss', d_out),
                
                tf.summary.scalar('misc/measure', measure),
                tf.summary.scalar('misc/k_t', k_t),
                tf.summary.scalar('misc/g_lr', g_lr),
                tf.summary.scalar('misc/r_lr', r_lr),
                tf.summary.scalar('misc/d_lr', d_lr),
                tf.summary.scalar('misc/balance', balance),
            ])

            summary_op = tf.summary.merge(summary_set)

            savers = {}
            for subgraph, variables in saver_pairs:
                savers[subgraph] = tf.train.Saver(sess.graph.get_collection(variables))

            conngraph.add_subgraph_savers(savers)

        tf.add_to_collection('step', step)
        
        sess.graph.clear_collection('outputs')
        sess.graph.clear_collection('outputs')
        for g_loss in g_losses:
            tf.add_to_collection('outputs_interim', g_loss)
        for r_loss in r_losses:
            tf.add_to_collection('outputs_interim', r_loss)
        tf.add_to_collection('outputs_interim', d_out)
        tf.add_to_collection('outputs_interim', k_t)
        tf.add_to_collection('outputs_interim', summary_op)
        tf.add_to_collection('outputs', k_update)
        tf.add_to_collection('outputs', measure)
        tf.add_to_collection('outputs_lr', g_lr_update)
        tf.add_to_collection('outputs_lr', r_lr_update)
        tf.add_to_collection('outputs_lr', d_lr_update)
        tf.add_to_collection('summary', summary_op)

    return conngraph


def build_feed_func(gen_tensor, gen_input, rev_inputs, data_inputs, alpha_tensor, **keys):
    def get_feed_dict(self, trainer):

        config = trainer.c_graph.config

        step = trainer.sess.run(trainer.step)
        
        #feed_dict setup
        z = np.random.uniform(-1, 1, size=(trainer.batch_size, trainer.z_num))
        feeds = [
            (gen_input, z), 
        ]
        
        x = trainer.data_loader
        x = trainer.sess.run(x)
        x = norm_img(x) #running numpy version so don't have to modify graph
        for inpt in data_inputs:
            feeds.append((inpt, x))

        if config.alphas:
            self.alphas_feed = [[alpha_tensor.format(i), 1.0] for i in range(config.repeat_num-1)]
            if step < config.alpha_update_steps:
                self.alphas_feed[trainer.c_graph.block_index][1] = 0.0
            elif step >= config.alpha_update_steps*2:
                self.alphas_feed[trainer.c_graph.block_index][1] = 1.0
            else:
                val = (step-config.alpha_update_steps)/float(config.alpha_update_steps)
                self.alphas_feed[trainer.c_graph.block_index][1] = val
            feeds.extend(self.alphas_feed)

        reverse = [(rinput, np.zeros([trainer.batch_size, 3, trainer.img_size, trainer.img_size]))
                   for rinput in rev_inputs]
        gen_outputs = [trainer.sess.run(gtensor, dict(feeds+reverse)) for gtensor in gen_tensor]
        feeds.extend([(rev_inputs[i], gen_outputs[i]) for i in range(len(rev_inputs))])
        
        feed_dict = dict(feeds)
        return feed_dict

    return get_feed_dict


def build_send_func(gen_input, rev_inputs, data_inputs, gen_outputs, a_output, **keys):
    def send_outputs(self, trainer, step):
        if not hasattr(self, 'z_fixed'):
            self.z_fixed = np.random.uniform(-1, 1, size=(trainer.batch_size, trainer.z_num))
            self.x_fixed = trainer.get_image_from_loader()
            save_image(self.x_fixed, os.path.join(trainer.log_dir, 'x_fixed.png'))
            self.x_fixed = norm_img(self.x_fixed)

        alphas = trainer.c_graph.config.alphas

        #generate
        z_fixed = self.z_fixed
        feeds = [(gen_input, z_fixed)]
        if alphas:
            feeds.extend(self.alphas_feed)
        reverse = [(r_inpt, np.zeros([trainer.batch_size, 3, trainer.img_size, trainer.img_size])) for r_inpt in rev_inputs]
        g_outputs = [trainer.sess.run(g_out, dict(feeds+reverse)) for g_out in  gen_outputs['G']]
        for i, r_inpt in enumerate(rev_inputs):
            feeds.append((r_inpt, g_outputs[i]))
        feeds = dict(feeds)

        for name, output in gen_outputs.items():
            x_gen = trainer.sess.run(trainer.sess.graph.get_tensor_by_name(output), feeds)
            if name == 'G':
                gen = x_gen
            save_image(denorm_img_numpy(x_gen, trainer.data_format),
                       os.path.join(trainer.log_dir, '{}_{}.png'.format(step, name)))

        #autoencode
        for k, img in (('real', self.x_fixed), ('gen', gen)):
            if img is None:
                continue
            if img.shape[3] in [1, 3]:
                img = img.transpose([0, 3, 1, 2])
            afeeds = [
                (gen_input, z_fixed),
            ]
            for inpt in data_inputs:
                afeeds.append((inpt, img))
            if alphas:
                afeeds.extend(self.alphas_feed)
            afeeds = dict(afeeds+reverse)
            x = trainer.sess.run(trainer.sess.graph.get_tensor_by_name(a_output),
                                 afeeds)
            save_image(denorm_img_numpy(x, trainer.data_format),
                       os.path.join(trainer.log_dir, '{}_A_{}.png'.format(step, k)))

        #interpolate
        z_flex = np.random.uniform(-1, 1, size=(trainer.batch_size, trainer.z_num))

        for i, g_out in  enumerate(gen_outputs['G']):
            generated = []
            for _, ratio in enumerate(np.linspace(0, 1, 10)):
                z = np.stack([slerp(ratio, r1, r2) for r1, r2 in zip(z_fixed, z_flex)])
                #generate
                feeds = [(gen_input, z)]
                if alphas:
                    feeds.extend(self.alphas_feed)
                feeds = dict(feeds+reverse)
                z_decode = trainer.sess.run(trainer.sess.graph.get_tensor_by_name(g_out),
                                            feeds)
                generated.append(denorm_img_numpy(z_decode, trainer.data_format))

            generated = np.stack(generated).transpose([1, 0, 2, 3, 4])

            all_img_num = np.prod(generated.shape[:2])
            batch_generated = np.reshape(generated, [all_img_num] + list(generated.shape[2:]))
            save_image(batch_generated, os.path.join(trainer.log_dir, '{}_interp_G{}.png'.format(step, i)), nrow=10)

    return send_outputs
