from models.models import *
from models.nvd_cg.graph import *
from models.nvd_cg.config import *
import numpy as np
import tensorflow as tf


def test():
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        z = tf.placeholder(tf.float32, [None, 128])
        output_num = 3
        repeat_num = 4
        hidden_num = 128
        z_num = 128
        data_format = 'NCHW'
        alphas = [tf.Variable(0., name='alpha'+str(i)) for i in range(repeat_num-1)]
        out_g, vg = GeneratorSkipCNN(z, hidden_num, output_num, repeat_num, alphas, data_format, False)
        out_d, z, vd = DiscriminatorSkipCNN(out_g, output_num, z_num, repeat_num, hidden_num, data_format)
        # out_d = out_g[::-1]
        # for i in range(1, len(out_d)):
        #     out_d[i] = slim.conv2d(out_d[i], hidden_num*i, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        # out_d[0] = slim.conv2d(out_d[0], hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
    return graph, out_g, out_d


def test_c():
    return build_graph(config('began_b16_z128_sz64_h128_g0.5'))


def test_c2():
    conf = config('began_b16_z128_sz64_h128_g0.5')
    c = build_graph(conf)
    with tf.Session(graph=tf.Graph()) as sess:
        c.connect_graph(inputs, outputs, sess)

        # alphas = []
        # for i in range(conf.repeat_num-1):
        #     alpha = c.get_variable(GENR+ALPH.format(i))
        #     alphas.append(tf.identity(alpha, name='alpha'+str(i)))

        d_losses = []
        g_losses = []
        for i in range(conf.repeat_num):
            d_loss = sess.graph.get_tensor_by_name(LSDN.format(i)+OUTP)
            g_loss = sess.graph.get_tensor_by_name(LSGN.format(i)+OUTP)
            d_loss = tf.identity(d_loss, name='d_loss'+str(i))
            g_loss = tf.identity(g_loss, name='g_loss'+str(i)) #collapsing to NaN
            d_losses.append(d_loss)
            g_losses.append(g_loss)

        d_mean = tf.reduce_mean(tf.stack(d_losses))
        g_mean = tf.reduce_mean(tf.stack(g_losses))

        g_lr = tf.Variable(conf.g_lr, name='g_lr')
        d_lr = tf.Variable(conf.d_lr, name='d_lr')

        g_lr_update = tf.assign(g_lr, tf.maximum(g_lr * 0.5, conf.lr_lower_boundary), name='g_lr_update')
        d_lr_update = tf.assign(d_lr, tf.maximum(d_lr * 0.5, conf.lr_lower_boundary), name='d_lr_update')

        g_optimizer = tf.train.AdamOptimizer(g_lr)
        d_optimizer = tf.train.AdamOptimizer(d_lr)

        g_optim = g_optimizer.minimize(g_mean, var_list=tf.get_collection(GENR+VARS))
        d_optim = d_optimizer.minimize(d_mean, var_list=tf.get_collection(DISC+VARS))

        # bound = 0.1
        # conds = [(1-alphas[0]) > bound]
        # conds.extend([tf.logical_and(alphas[i] > bound, alphas[i+1] > bound)  for i in range(len(alphas) - 1)])
        # conds.append(alphas[-1] > bound)
        # controls = [tf.cond(conds[i], lambda: g_optims[i], lambda: tf.no_op())
        #             for i in range(conf.repeat_num)]

        z = np.random.uniform(-1, 1, [16, 128])
        feeds = [
            (GENR+INPT, z),
        ]
        for i in range(conf.repeat_num):
            feeds.append((CNCN.format(i)+D_IN, np.random.uniform(-1, 1, [16, 3, 2**(3+i), 2**(3+i)])))
            feeds.append((LSDN.format(i)+O_IN, np.random.uniform(-1, 1, [16, 3, 2**(3+i), 2**(3+i)])))
        alphas = []
        for i in range(conf.repeat_num-1):
            val = 1 - conf.alpha_update_steps*(2*(i+1) - 1)
            if val < 0:
                num = 0.1
            else:
                num = np.min([0.1 + (val // conf.alpha_update_step_size), 0.9])
            alphas.append((GENR+ALPH.format(i), num))
        feeds.extend(alphas)
        feeds = dict(feeds)
    return c, feeds, alphas, g_mean, d_mean, g_optim, d_optim
