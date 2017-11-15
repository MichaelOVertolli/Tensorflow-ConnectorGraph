from ..connectorgraph import ConnectorGraph
from ..errors import FirstInitialization
import numpy as np
import os
from skimage.measure import block_reduce
from ..subgraph import BuiltSubGraph, SubGraph
import tensorflow as tf
from ..model_utils import *


#Models
GENR = 'nvd_generator_0'
DISC = 'nvd_discriminator_0'
LSGN = 'cqs_loss_set_1{}'
LSDN = 'cqs_loss_set_2{}'
CNCN = 'cqs_concat_{}'
SPLN = 'cqs_split_{}'
CQST = 'nvd_train'

#Inputs
INPT = '/input:0'
INPN = '/input{}:0'
G_IN = '/gen_input:0'
D_IN = '/data_input:0'
O_IN = '/orig_input:0'
A_IN = '/autoencoded_input:0'

#Outputs
OUTP = '/output:0'
OUTN = '/output{}:0'
GOUT = '/gen_output:0'
DOUT = '/data_output:0'

#Alphas
ALPH = '/alpha{}:0'

#Variables
VARS = '/trainable_variables'

connections = []

inputs = [
    GENR+INPT,
]

outputs = []


def build_graph(config):
    #TODO: fix partial loading of saved variables from this model into partial models
    generator = init_subgraph(GENR, config.mdl_type)
    discriminator = init_subgraph(DISC, config.mdl_type)

    conngraph = ConnectorGraph(config)
    conngraph.add_subgraph(generator)
    conngraph.add_subgraph(discriminator)
    
    for i in range(config.repeat_num):
        gloss = init_subgraph(LSGN.format(i), config.lss_type)
        dloss = init_subgraph(LSDN.format(i), config.lss_type)
        concat_op = init_subgraph(CNCN.format(i), '')
        split_op = init_subgraph(SPLN.format(i), '')
        conngraph.add_subgraph(gloss)
        conngraph.add_subgraph(dloss)
        conngraph.add_subgraph(concat_op)
        conngraph.add_subgraph(split_op)

        conns = [
            [GENR,           CNCN.format(i), GENR+OUTN.format(i), CNCN.format(i)+G_IN],
            [CNCN.format(i), DISC,           CNCN.format(i)+OUTP, DISC+INPN.format(i)],
            [DISC,           SPLN.format(i), DISC+OUTN.format(i), SPLN.format(i)+INPT],
            [GENR,           LSGN.format(i), GENR+OUTN.format(i), LSGN.format(i)+O_IN],
            [SPLN.format(i), LSGN.format(i), SPLN.format(i)+GOUT, LSGN.format(i)+A_IN],
            [SPLN.format(i), LSDN.format(i), SPLN.format(i)+DOUT, LSDN.format(i)+A_IN],
        ]
        connections.extend(conns)
        inpts = [
            CNCN.format(i)+D_IN,
            LSDN.format(i)+O_IN, #same as CNCN.format(i)+D_IN
        ]
        inputs.extend(inpts)
        out = [
            LSGN.format(i)+OUTP,
            LSDN.format(i)+OUTP,
        ]
        outputs.extend(out)

    conngraph.print_subgraphs()

    conngraph.quick_connect(connections)



    with tf.Session(graph=tf.Graph()) as sess:
        conngraph.connect_graph(inputs, outputs, sess)

        step = tf.Variable(0, name='step', trainable=False)
        
        with tf.variable_scope('cqs_train'):
            d_losses = []
            g_losses = []
            for i in range(config.repeat_num):
                d_loss = sess.graph.get_tensor_by_name(LSDN.format(i)+OUTP)
                g_loss = sess.graph.get_tensor_by_name(LSGN.format(i)+OUTP)
                d_loss = tf.identity(d_loss, name='d_loss'+str(i))
                g_loss = tf.identity(g_loss, name='g_loss'+str(i))
                d_losses.append(d_loss)
                g_losses.append(g_loss)

            d_mean = tf.reduce_mean(tf.stack(d_losses))
            g_mean = tf.reduce_mean(tf.stack(g_losses))
            k_t = tf.Variable(0., trainable=False, name='k_t')

            d_out = d_mean - k_t * g_mean
            d_out = tf.identity(d_out, name='d_loss_out')
            g_out = tf.identity(g_mean, name='g_loss_out')

            g_lr = tf.Variable(config.g_lr, name='g_lr')
            d_lr = tf.Variable(config.d_lr, name='d_lr')

            g_lr_update = tf.assign(g_lr, tf.maximum(g_lr * 1.0, config.lr_lower_boundary), name='g_lr_update')
            d_lr_update = tf.assign(d_lr, tf.maximum(d_lr * 1.0, config.lr_lower_boundary), name='d_lr_update')

            g_optimizer = tf.train.AdamOptimizer(g_lr)
            d_optimizer = tf.train.AdamOptimizer(d_lr)

            g_optim = g_optimizer.minimize(g_mean, global_step=step, var_list=tf.get_collection(GENR+VARS))
            d_optim = d_optimizer.minimize(d_out, var_list=tf.get_collection(DISC+VARS))

            balance = config.gamma * d_mean - g_mean
            measure = d_mean + tf.abs(balance)
            measure = tf.identity(measure, name='measure')

            with tf.control_dependencies([d_optim, g_optim]):
                k_update = tf.assign(k_t, tf.clip_by_value(k_t + config.lambda_k * balance, 0, 1))
                k_update = tf.identity(k_update, name='k_update')

            summaries = [
                tf.summary.scalar('loss/d_out', d_out),
                tf.summary.scalar('loss/g_out', g_out),

                tf.summary.scalar('misc/measure', measure),
                tf.summary.scalar('misc/k_t', k_t),
                tf.summary.scalar('misc/g_lr', g_lr),
                tf.summary.scalar('misc/d_lr', d_lr),
                tf.summary.scalar('misc/balance', balance),
            ]
            for i in range(config.repeat_num):
                summ = [
                    tf.summary.image('G_'+str(8*(2**i)), denorm_img(sess.graph.get_tensor_by_name(GENR+OUTN.format(i)), config.data_format)),
                    tf.summary.image('AE_G_'+str(8*(2**i)), denorm_img(sess.graph.get_tensor_by_name(SPLN.format(i)+GOUT), config.data_format)),
                    tf.summary.image('AE_D_'+str(8*(2**i)), denorm_img(sess.graph.get_tensor_by_name(SPLN.format(i)+DOUT), config.data_format)),

                    tf.summary.scalar('loss/g_loss_'+str(8*(2**i)), g_losses[i]),
                    tf.summary.scalar('loss/d_loss_'+str(8*(2**i)), d_losses[i]),
                ]
                summaries.extend(summ)
            for i in range(config.repeat_num-1):
                summaries.append(tf.summary.scalar('misc/alpha_'+str(i), sess.graph.get_tensor_by_name(GENR+ALPH.format(i))))
            
            summary_op = tf.summary.merge(summaries)

            savers = {
                GENR: tf.train.Saver(sess.graph.get_collection('/'.join([GENR, 'variables']))),
                DISC: tf.train.Saver(sess.graph.get_collection('/'.join([DISC, 'variables'])))
            }

            conngraph.add_subgraph_savers(savers)

        tf.add_to_collection('step', step)
        
        sess.graph.clear_collection('outputs')
        tf.add_to_collection('outputs_interim', d_out)
        tf.add_to_collection('outputs_interim', g_out)
        tf.add_to_collection('outputs_interim', k_t)
        tf.add_to_collection('outputs_interim', summary_op)
        tf.add_to_collection('outputs', k_update)
        tf.add_to_collection('outputs', measure)
        tf.add_to_collection('outputs_lr', g_lr_update)
        tf.add_to_collection('outputs_lr', d_lr_update)
        tf.add_to_collection('summary', summary_op)

        def get_feed_dict(self, trainer):
            x = trainer.data_loader
            x = trainer.sess.run(x)
            x = norm_img(x) #running numpy version so don't have to modify graph
            z = np.random.uniform(-1, 1, size=(trainer.batch_size, trainer.z_num))
            feeds = [
                (GENR+INPT, z),
            ]
            for i in range(config.repeat_num):
                block_size = 2**(config.repeat_num - i - 1) #smallest size first
                img = block_reduce(x, (1, 1, block_size, block_size), np.mean)
                feeds.append((CNCN.format(i)+D_IN, img))
                feeds.append((LSDN.format(i)+O_IN, img))
            step = trainer.sess.run(trainer.step)
            self.alphas_feed = []
            for i in range(config.repeat_num-1):
                val = step - config.alpha_update_steps*(2*(i+1) - 1)
                if val < 0:
                    num = 0.1
                else:
                    num = np.min([0.1 + (val // config.alpha_update_step_size)*0.1, 0.9])
                self.alphas_feed.append((GENR+ALPH.format(i), num))
            feeds.extend(self.alphas_feed)
            feed_dict = dict(feeds)
            return feed_dict
        
        conngraph.attach_func(get_feed_dict)
        
        def send_outputs(self, trainer, step):
            if not hasattr(self, 'z_fixed'):
                self.z_fixed = np.random.uniform(-1, 1, size=(trainer.batch_size, trainer.z_num))
                self.x_fixed = trainer.get_image_from_loader()
                save_image(self.x_fixed, os.path.join(trainer.log_dir, 'x_fixed.png'))
                self.x_fixed = norm_img(self.x_fixed)
                img = self.x_fixed.copy()
                if img.shape[3] in [1, 3]:
                    img = img.transpose([0, 3, 1, 2])
                feeds = [
                    (GENR+INPT, self.z_fixed),
                ]
                for j in range(config.repeat_num):
                    block_size = 2**(config.repeat_num - j - 1) #smallest size first
                    img_ = block_reduce(img, (1, 1, block_size, block_size), np.mean)
                    feeds.append((CNCN.format(j)+D_IN, img_))
                self.feed_dict_ = dict(feeds)

            i = np.min([step // int(config.alpha_update_steps*2), config.repeat_num-1])

            #generate
            feeds = [(GENR+INPT, self.z_fixed)]
            feeds.extend(self.alphas_feed)
            feeds = dict(feeds)
            x_gens = trainer.sess.run([trainer.sess.graph.get_tensor_by_name(GENR+OUTN.format(j))
                                       for j in range(config.repeat_num)],
                                      feeds)
            
            save_image(denorm_img_numpy(x_gens[i], trainer.data_format),
                       os.path.join(trainer.log_dir, '{}_G.png'.format(step)))

            #autoencode
            for k, imgs in (('real', [self.x_fixed]), ('gen', x_gens)):
                if imgs is None:
                    continue
                for i, img in enumerate(imgs):
                    if img.shape[3] in [1, 3]:
                        imgs[i] = img.transpose([0, 3, 1, 2])
                feed_dict = [_ for _ in self.feed_dict_.items()]
                feed_dict.extend(self.alphas_feed)
                feed_dict = dict(feed_dict)
                if k == 'gen':
                    for j in range(config.repeat_num):
                        feed_dict[CNCN.format(j)+D_IN] = imgs[j]
                x = trainer.sess.run(trainer.sess.graph.get_tensor_by_name(SPLN.format(i)+DOUT),
                                     feed_dict)
                save_image(denorm_img_numpy(x, trainer.data_format),
                           os.path.join(trainer.log_dir, '{}_D_{}.png'.format(step, k)))


            #interpolate
            z_flex = np.random.uniform(-1, 1, size=(trainer.batch_size, trainer.z_num))
            generated = []
            for _, ratio in enumerate(np.linspace(0, 1, 10)):
                z = np.stack([slerp(ratio, r1, r2) for r1, r2 in zip(self.z_fixed, z_flex)])
                #generate
                feeds = [(GENR+INPT, z)]
                feeds.extend(self.alphas_feed)
                feeds = dict(feeds)
                z_decode = trainer.sess.run(trainer.sess.graph.get_tensor_by_name(GENR+OUTN.format(i)),
                                            feeds)
                generated.append(denorm_img_numpy(z_decode, trainer.data_format))

            generated = np.stack(generated).transpose([1, 0, 2, 3, 4])

            all_img_num = np.prod(generated.shape[:2])
            batch_generated = np.reshape(generated, [all_img_num] + list(generated.shape[2:]))
            save_image(batch_generated, os.path.join(trainer.log_dir, '{}_interp_G.png'.format(step)), nrow=10)

        conngraph.attach_func(send_outputs)
        
    return conngraph
