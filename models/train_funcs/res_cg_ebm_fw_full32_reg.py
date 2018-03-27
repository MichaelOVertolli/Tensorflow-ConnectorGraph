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
from ..errors import FirstInitialization, ModalCollapseError
from ..graphs.converter import build_variables
import numpy as np
import os
from skimage.measure import block_reduce
from scipy.ndimage import zoom
from ..subgraph import BuiltSubGraph, SubGraph
import tensorflow as tf
from ..model_utils import *
import time


def build_train_ops(log_dir, conngraph, inputs, outputs,
                    train_scope, loss_tensors, train_sets,
                    img_pairs, saver_pairs, alpha_tensor, **keys):
    config = conngraph.config
    with tf.Session(graph=tf.Graph()) as sess:
        conngraph.connect_graph(inputs, outputs, sess)

        variables = build_variables(conngraph, sess, train_sets)
        
        step = tf.Variable(0, dtype=tf.int32, name='step', trainable=False)
        
        with tf.variable_scope(train_scope) as vs:

            k_t = tf.Variable(0., trainable=False, name='k_t')
            lm_gms = tf.Variable(config.lambda_gms, trainable=False, name='lambda_gms')
            lm_chrom = tf.Variable(config.lambda_chrom, trainable=False, name='lambda_chrom')

            g_losses = sess.graph.get_tensor_by_name(loss_tensors['G'])
            g_l1 = tf.identity(g_losses[0], name='g_l1')
            g_gms = tf.identity(g_losses[1], name='g_gms')
            g_chrom = tf.identity(g_losses[2], name='g_chrom')
            
            g_loss = g_l1 + g_gms*lm_gms + g_chrom*lm_chrom
            g_loss = tf.identity(g_loss, name='g_loss')

            d_losses = sess.graph.get_tensor_by_name(loss_tensors['D'])
            d_l1 = tf.identity(d_losses[0], name='d_l1')
            d_gms = tf.identity(d_losses[1], name='d_gms')
            d_chrom = tf.identity(d_losses[2], name='d_chrom')

            d_loss = d_l1 + d_gms*lm_gms + d_chrom*lm_chrom
            d_loss = tf.identity(d_loss, name='d_raw_loss')
            d_out = d_loss - k_t * g_loss
            d_out = tf.identity(d_out, name='d_loss')

            g_lr = tf.Variable(config.g_lr, name='g_lr')
            d_lr = tf.Variable(config.d_lr, name='d_lr')

            g_lr_update = tf.assign(g_lr, tf.maximum(g_lr * 0.5, config.lr_lower_boundary), name='g_lr_update')
            d_lr_update = tf.assign(d_lr, tf.maximum(d_lr * 0.5, config.lr_lower_boundary), name='d_lr_update')

            g_optimizer = tf.train.AdamOptimizer(g_lr)
            d_optimizer = tf.train.AdamOptimizer(d_lr)

            # TODO: add incremental variable training as separate mod
            g_optim = g_optimizer.minimize(g_loss, global_step=step, var_list=variables['G'])
            d_optim = d_optimizer.minimize(d_out, var_list=variables['D'])

            balance = config.gamma * d_loss - g_loss
            measure = d_loss + tf.abs(balance)
            measure = tf.identity(measure, name='measure')

            with tf.control_dependencies([d_optim, g_optim]):
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

            summary_set.extend([
                tf.summary.scalar('loss/g_loss', g_loss),
                tf.summary.scalar('loss/g_l1', g_l1),
                tf.summary.scalar('loss/g_gms', g_gms),
                tf.summary.scalar('loss/g_chrom', g_chrom),
                tf.summary.scalar('loss/d_raw_loss', d_loss),
                tf.summary.scalar('loss/d_l1', d_l1),
                tf.summary.scalar('loss/d_gms', d_gms),
                tf.summary.scalar('loss/d_chrom', d_chrom),
                tf.summary.scalar('loss/d_loss', d_out),          
                tf.summary.scalar('misc/measure', measure),
                tf.summary.scalar('misc/k_t', k_t),
                tf.summary.scalar('misc/g_lr', g_lr),
                tf.summary.scalar('misc/d_lr', d_lr),
                tf.summary.scalar('misc/balance', balance),
            ])

            summary_op = tf.summary.merge(summary_set)

            savers = {}
            for subgraph, variables in saver_pairs:
                saver = tf.train.Saver(sess.graph.get_collection(variables))
                savers[subgraph] = saver.as_saver_def()

            conngraph.add_subgraph_savers(savers)

        tf.add_to_collection('step', step)
        
        sess.graph.clear_collection('outputs')
        sess.graph.clear_collection('outputs')
        tf.add_to_collection('outputs_interim', g_loss)
        tf.add_to_collection('outputs_interim', d_loss)
        tf.add_to_collection('outputs_interim', d_out)
        tf.add_to_collection('outputs_interim', k_t)
        tf.add_to_collection('outputs_interim', summary_op)
        tf.add_to_collection('outputs', k_update)
        tf.add_to_collection('outputs', measure)
        tf.add_to_collection('outputs_lr', g_lr_update)
        tf.add_to_collection('outputs_lr', d_lr_update)
        tf.add_to_collection('summary', summary_op)

        sess.run(step.initializer)
        sess.run(tf.variables_initializer(tf.contrib.framework.get_variables(vs)))

        full_saver = tf.train.Saver()
        saver_dir = os.path.join(log_dir, 'temp')
        if not os.path.exists(saver_dir):
            os.makedirs(saver_dir)
        saver_dir = os.path.join(saver_dir,'temp')
        conngraph.set_init_params(full_saver, saver_dir)
        
        full_saver.save(sess, saver_dir)

    return conngraph


def build_feed_func(gen_tensor, gen_input, rev_input, data_inputs, alpha_tensor, **keys):
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
        # x = norm_img(x) #running numpy version so don't have to modify graph
        for inpt in data_inputs:
            feeds.append((inpt, x))

        feed_dict = dict(feeds)
        return feed_dict

    return get_feed_dict


def build_send_func(gen_input, rev_input, data_inputs, gen_outputs, a_output, **keys):
    def send_outputs(self, trainer, step):
        if not hasattr(self, 'z_fixed'):
            self.z_fixed = np.random.uniform(-1, 1, size=(trainer.batch_size, trainer.z_num))
            self.x_fixed = trainer.get_image_from_loader()
            save_image(self.x_fixed, os.path.join(trainer.log_dir, 'x_fixed.png'))
            # self.x_fixed = norm_img(self.x_fixed)

        #generate
        z_fixed = self.z_fixed
        feeds = [(gen_input, z_fixed)]
        feeds = dict(feeds)

        for name, output in gen_outputs.items():
            x_gen = trainer.sess.run(trainer.sess.graph.get_tensor_by_name(output), feeds)
            # if np.mean(np.absolute(x_gen[0, :, :, :] - x_gen[15, :, :, :])) < 0.0005:
            #     raise ModalCollapseError()
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
            afeeds = dict(afeeds)
            x = trainer.sess.run(trainer.sess.graph.get_tensor_by_name(a_output),
                                 afeeds)
            save_image(denorm_img_numpy(x, trainer.data_format),
                       os.path.join(trainer.log_dir, '{}_A_{}.png'.format(step, k)))

        #interpolate
        z_flex = np.random.uniform(-1, 1, size=(trainer.batch_size, trainer.z_num))
        generated = []
        for _, ratio in enumerate(np.linspace(0, 1, 10)):
            z = np.stack([slerp(ratio, r1, r2) for r1, r2 in zip(z_fixed, z_flex)])
            #generate
            feeds = [(gen_input, z)]
            feeds = dict(feeds)
            z_decode = trainer.sess.run(trainer.sess.graph.get_tensor_by_name(gen_outputs['G']),
                                        feeds)
            generated.append(denorm_img_numpy(z_decode, trainer.data_format))

        generated = np.stack(generated).transpose([1, 0, 2, 3, 4])

        all_img_num = np.prod(generated.shape[:2])
        batch_generated = np.reshape(generated, [all_img_num] + list(generated.shape[2:]))
        save_image(batch_generated, os.path.join(trainer.log_dir, '{}_interp_G.png'.format(step)), nrow=10)

    return send_outputs
