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

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer as var_init
slim = tf.contrib.slim


def ResidualBlock(inpt,
                  hidden_num,
                  size,
                  activation_fn,
                  block_name,
                  alphas={'residual':1.0,
                          'shortcut':1.0},
                  project=True, # True if hidden_num changes between blocks
                  resample=None, #up, down, None
                  pool_type='conv', #conv, avg_pool
                  weights_initializer=tf.contrib.layers.xavier_initializer(),
                  normalizer_fn=None,
                  normalizer_params=None,
                  reuse=False,
                  data_format='NCHW'):
    with tf.variable_scope(block_name, reuse=reuse) as vs:
        if project:
            fc_size = np.prod([hidden_num, size, size])
            shortcut = slim.fully_connected(inpt, fc_size, activation_fn=None) # linear projection
            shortcut = reshape(shortcut, hidden_num, size, size, data_format)
        else:
            shortcut = inpt
        x = slim.conv2d(inpt, hidden_num, 3, 1, activation_fn=activation_fn,
                        weights_initializer=weights_initializer,
                        normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
                        data_format=data_format)
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=activation_fn,
                        weights_initializer=weights_initializer,
                        normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
                        data_format=data_format)
        if resample == 'up':
            x = upscale(x, 2, data_format)
            shortcut = upscale(shortcut, 2, data_format)
        elif resample == 'down':
            if pool_type == 'conv':
                x = slim.conv2d(x, hidden_num, 3, 2, activation_fn=activation_fn, 
                                weights_initializer=weights_initializer,
                                normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
                                data_format=data_format)
                shortcut = slim.conv2d(shortcut, hidden_num, 3, 2, activation_fn=activation_fn, 
                                       weights_initializer=weights_initializer,
                                       normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
                                       data_format=data_format)
            elif pool_type == 'avg_pool':
                x = tf.nn.avg_pool(x, [1, 1, 2, 2], [1, 1, 1, 1], 'VALID', data_format=data_format)
                shortcut = tf.nn.avg_pool(shortcut, [1, 1, 2, 2], [1, 1, 1, 1], 'VALID', data_format=data_format)
        x = alphas['residual']*x + alphas['shortcut']*shortcut
    variables = tf.contrib.framework.get_variables(vs)
    return x, variables


def alphas2dict(alphas, i):
    if alphas is None:
        out = {'residual': 1.0, 'shortcut':1.0}
    else:
        out = {'residual': alphas[i], 'shortcut':1-alphas[i]}
    return out


def blockname(resample, i, total):
    name = 'RB_'
    if resample == 'up' or resample is None:
        name += str(i)
    else:
        name += str(total - 2 - i)
    return name


def ResNet(inpt,
           repeat_num,
           hidden_nums, #assumes the inc/decr mirrors resample type
           sizes, #assumes the inc/decr mirrors resample type
           activation_fn,
           net_name,
           alphas=None, # None or list of float in [0.0, 1.0]
           project=True,
           minibatch=False,
           resample=None, #up, down, None
           pool_type='conv', #conv, avg_pool
           weights_initializer=tf.contrib.layers.xavier_initializer(),
           normalizer_fn=None,
           normalizer_params=None,
           reuse=False,
           data_format='NCHW'):
    sep_variables = {}
    with tf.variable_scope(net_name) as vs:
        if resample == 'up':
            with tf.variable_scope('front', reuse=reuse) as vs_front:
                fc_size = np.prod([hidden_nums[0], sizes[0], sizes[0]])
                x = slim.fully_connected(inpt, fc_size, activation_fn=None)
                x = reshape(x, sizes[0], sizes[0], hidden_nums[0], data_format)
            sep_variables['front'] = tf.contrib.framework.get_variables(vs_front)
        else:
            x = inpt
        for i in range(repeat_num):
            if i < repeat_num-1:
                block_name = blockname(resample, i, repeat_num)
                alpha = alphas2dict(alphas, i)
                x, v = ResidualBlock(x, hidden_nums[i], sizes[i], activation_fn, block_name,
                                     alpha, project, resample, pool_type,
                                     weights_initializer, normalizer_fn, normalizer_params,
                                     reuse, data_format)
                sep_variables[block_name] = v
            else:
                with tf.variable_scope('end', reuse=reuse) as vs_end:
                    if minibatch:
                        x = minibatch_disc_concat(x)
                    x = slim.conv2d(x, hidden_nums[i], 3, 1, activation_fn=activation_fn, 
                                    weights_initializer=weights_initializer,
                                    normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
                                    data_format=data_format)
                    if resample == 'up':
                        x = slim.conv2d(x, hidden_nums[i], 3, 1, activation_fn=activation_fn,  
                                        weights_initializer=weights_initializer,
                                        normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
                                        data_format=data_format)
                        x = slim.conv2d(x, 3, 3, 1, activation_fn=None, data_format=data_format)
                    elif resample == 'down': #I think this is wrong.... hidden_nums should be z_num (not using though)
                        x = slim.conv2d(x, hidden_nums[i], 4, 1, padding='VALID', activation_fn=activation_fn, 
                                        weights_initializer=weights_initializer,
                                        normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
                                        data_format=data_format)
                        x = tf.squeeze(x, [2, 3])
                sep_variables['end'] = tf.contrib.framework.get_variables(vs_end)
    all_variables = tf.contrib.framework.get_variables(vs)
    return x, all_variables, sep_variables

def GeneratorCNN(z, hidden_num, output_num, repeat_num, data_format, reuse):
    with tf.variable_scope("G", reuse=reuse) as vs:
        num_output = int(np.prod([8, 8, hidden_num]))
        x = slim.fully_connected(z, num_output, activation_fn=None)
        x = reshape(x, 8, 8, hidden_num, data_format)
        
        for idx in range(repeat_num):
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            if idx < repeat_num - 1:
                x = upscale(x, 2, data_format)

        out = slim.conv2d(x, 3, 3, 1, activation_fn=None, data_format=data_format)

    variables = tf.contrib.framework.get_variables(vs)
    return out, variables


def GeneratorRCNN(x, input_channel, z_num, repeat_num, hidden_num, data_format):
    with tf.variable_scope("GR") as vs:
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, #normalizer_fn=slim.batch_norm,
                        data_format=data_format)

        prev_channel_num = hidden_num
        for idx in range(repeat_num):
            channel_num = hidden_num * (idx + 1)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, #normalizer_fn=slim.batch_norm,
                            data_format=data_format)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, #normalizer_fn=slim.batch_norm,
                            data_format=data_format)
            if idx < repeat_num - 1:
                x = slim.conv2d(x, channel_num, 3, 2, activation_fn=tf.nn.elu, #normalizer_fn=slim.batch_norm,
                                data_format=data_format)

        x = tf.reshape(x, [-1, np.prod([8, 8, channel_num])])
        z = slim.fully_connected(x, z_num, activation_fn=None)

    variables = tf.contrib.framework.get_variables(vs)
    return z, variables


def GeneratorNSkipCNN(z, hidden_num, output_num, repeat_num, alphas, data_format, reuse):
    with tf.variable_scope("G", reuse=reuse) as vs:
        x = tf.expand_dims(tf.expand_dims(z, 2), 3)
        x = tf.pad(x, [[0, 0], [0, 0], [3, 3], [3, 3]], 'CONSTANT')
        x = slim.conv2d(x, hidden_num, 4, 1, padding='VALID', activation_fn=leaky_relu, weights_initializer=var_init(),
                        data_format=data_format)
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=leaky_relu, weights_initializer=var_init(),
                        normalizer_fn=slim.unit_norm, normalizer_params={'dim':1, 'epsilon':1e-8}, data_format=data_format)
        last = slim.conv2d(upscale(x, 2, data_format), 3, 3, 1, activation_fn=None, data_format=data_format)*(1-alphas[0])
        x = slim.conv2d(x, 3, 3, 1, activation_fn=None, data_format=data_format) #no alpha here as there's no sum from prev block
        channel_num = hidden_num
        for idx in range(1, repeat_num):
            x = slim.conv2d(x, channel_num, 1, 1, activation_fn=leaky_relu, weights_initializer=var_init(),
                            data_format=data_format)
            x = upscale(x, 2, data_format)
            if idx > 3:
                channel_num /= 2
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=leaky_relu, weights_initializer=var_init(),
                            normalizer_fn=slim.unit_norm, normalizer_params={'dim':1, 'epsilon':1e-8}, data_format=data_format)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=leaky_relu, weights_initializer=var_init(),
                            normalizer_fn=slim.unit_norm, normalizer_params={'dim':1, 'epsilon':1e-8}, data_format=data_format)
            x_ = slim.conv2d(x, 3, 3, 1, activation_fn=None, data_format=data_format)*alphas[idx-1]
            x_ += last
            if idx != repeat_num-1: #don't do this on the last block
                last = slim.conv2d(upscale(x, 2, data_format), 3, 3, 1, activation_fn=None, data_format=data_format)*(1-alphas[idx])
            x = x_

    variables = tf.contrib.framework.get_variables(vs)
    return x, variables


def GeneratorSkipCNN(z, hidden_num, output_num, repeat_num, alphas, data_format, reuse):
    with tf.variable_scope("G", reuse=reuse) as vs:
        num_output = int(np.prod([8, 8, hidden_num]))
        x = slim.fully_connected(z, num_output, activation_fn=None)
        x = reshape(x, 8, 8, hidden_num, data_format)
        out_set = []
        for idx in range(repeat_num):
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, #weights_initializer=var_init(),
                            data_format=data_format)
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, #weights_initializer=var_init(),
                            data_format=data_format)
            if idx < repeat_num - 1:
                out_set.append(x*(1 - alphas[idx]))
                x = upscale(x*alphas[idx], 2, data_format)

        out_set.append(x)

        for i in range(len(out_set)):
            out_set[i] = slim.conv2d(out_set[i], 3, 3, 1, activation_fn=None, data_format=data_format)

    variables = tf.contrib.framework.get_variables(vs)
    return out_set, variables


def DiscriminatorCNN(x, input_channel, z_num, repeat_num, hidden_num, data_format):
    with tf.variable_scope("D") as vs:
        # Encoder
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)

        prev_channel_num = hidden_num
        for idx in range(repeat_num):
            channel_num = hidden_num * (idx + 1)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            if idx < repeat_num - 1:
                x = slim.conv2d(x, channel_num, 3, 2, activation_fn=tf.nn.elu, data_format=data_format)
                #x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID')

        x = tf.reshape(x, [-1, np.prod([8, 8, channel_num])])
        z = x = slim.fully_connected(x, z_num, activation_fn=None)

        # Decoder
        num_output = int(np.prod([8, 8, hidden_num]))
        x = slim.fully_connected(x, num_output, activation_fn=None)
        x = reshape(x, 8, 8, hidden_num, data_format)
        
        for idx in range(repeat_num):
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            if idx < repeat_num - 1:
                x = upscale(x, 2, data_format)

        out = slim.conv2d(x, input_channel, 3, 1, activation_fn=None, data_format=data_format)

    variables = tf.contrib.framework.get_variables(vs)
    return out, z, variables


def DiscriminatorSkipCNN(xs, input_channel, z_num, repeat_num, hidden_num, data_format):
    with tf.variable_scope("D") as vs:
        # Encoder
        xs_ = xs[::-1]
        for i in range(1, len(xs_)):
            xs_[i] = slim.conv2d(xs_[i], hidden_num*i, 3, 1, activation_fn=tf.nn.elu, #weights_initializer=var_init(),
                                 data_format=data_format)
        x = slim.conv2d(xs_[0], hidden_num, 3, 1, activation_fn=tf.nn.elu, #weights_initializer=var_init(),
                        data_format=data_format)
        prev_channel_num = hidden_num
        for idx in range(repeat_num):
            channel_num = hidden_num * (idx + 1)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, #weights_initializer=var_init(),
                            data_format=data_format)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, #weights_initializer=var_init(),
                            data_format=data_format)
            if idx < repeat_num - 1:
                x = slim.conv2d(x, channel_num, 3, 2, activation_fn=tf.nn.elu, #weights_initializer=var_init(),
                                data_format=data_format)
                x = x + xs_[idx+1]
                #x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID')

        # x = minibatch_disc_concat(x)
        
        # x = tf.reshape(x, [-1, np.prod([8, 8, channel_num+1])])
        x = tf.reshape(x, [-1, np.prod([8, 8, channel_num])])
        z = x = slim.fully_connected(x, z_num, activation_fn=None)

        # Decoder
        num_output = int(np.prod([8, 8, hidden_num]))
        x = slim.fully_connected(x, num_output, activation_fn=None)
        x = reshape(x, 8, 8, hidden_num, data_format)
        out_set = []
        for idx in range(repeat_num):
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, #weights_initializer=var_init(),
                            data_format=data_format)
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, #weights_initializer=var_init(),
                            data_format=data_format)
            out_set.append(x)
            if idx < repeat_num - 1:
                x = upscale(x, 2, data_format)

        for i in range(len(out_set)):
            out_set[i] = slim.conv2d(out_set[i], input_channel, 3, 1, activation_fn=None, data_format=data_format)

    variables = tf.contrib.framework.get_variables(vs)
    return out_set, z, variables


def DiscriminatorNSkipCNN2(x, sizes, alphas, repeat_num, hidden_num, data_format):
    with tf.variable_scope('D') as vs:
        for i in range(repeat_num):
            if i < repeat_num-1:
                x = DiscriminatorNSkipCNNBlock2(x, sizes[i], sizes[i+1], alphas[i], data_format)
            else:
                x = DiscriminatorNSkipCNNLastBlock2(x, hidden_num, data_format)
    variables = tf.contrib.framework.get_variables(vs)
    return x, variables


def DiscriminatorNSkipCNNLastBlock2(x, hidden_num, data_format):
    x = slim.conv2d(x, hidden_num, 1, 1, activation_fn=leaky_relu, weights_initializer=var_init(),
                    data_format=data_format)
    x = minibatch_disc_concat(x)
    x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=leaky_relu, weights_initializer=var_init(),
                    data_format=data_format)
    x = slim.conv2d(x, hidden_num, 4, 1, padding='VALID', activation_fn=leaky_relu, weights_initializer=var_init(),
                    data_format=data_format)
    x = slim.fully_connected(x, 1, activation_fn=None)
    return x


def DiscriminatorNSkipCNNBlock2(x, first_size, second_size, alpha, data_format):
    next_ = slim.conv2d(x, second_size, 3, 2, activation_fn=leaky_relu, weights_initializer=var_init(),
                        data_format=data_format)
    next_ = slim.conv2d(next_, second_size, 1, 1, activation_fn=leaky_relu, weights_initializer=var_init(),
                        data_format=data_format)*(1-alpha)
    x = slim.conv2d(x, first_size, 1, 1, activation_fn=leaky_relu, weights_initializer=var_init(),
                    data_format=data_format)
    x = slim.conv2d(x, first_size, 3, 1, activation_fn=leaky_relu, weights_initializer=var_init(),
                    data_format=data_format)
    x = slim.conv2d(x, second_size, 3, 1, activation_fn=leaky_relu, weights_initializer=var_init(),
                    data_format=data_format)
    x = slim.conv2d(x, second_size, 3, 2, activation_fn=leaky_relu, weights_initializer=var_init(),
                    data_format=data_format)
    # x = slim.avg_pool2d(x, 2, padding='VALID', data_format=data_format) 
    x = x*alpha + next_
    x = slim.conv2d(x, 3, 3, 1, activation_fn=None, data_format=data_format)
    return x


def DiscriminatorNSkipCNN(x, sizes, scopes, alphas, cur_scope, hidden_num, data_format):
    variables = []
    for i, scope in enumerate(scopes):
        if scope == cur_scope:
            reuse = False
        else:
            reuse = True
        if scope != 'D0':
            x, v = DiscriminatorNSkipCNNBlock(x, sizes[i], sizes[i+1], alphas[i], scope, data_format, reuse)
        else:
            x, v = DiscriminatorNSkipCNNLastBlock(x, hidden_num, scope, data_format, reuse)
        variables.extend(v)
    return x, variables


def DiscriminatorNSkipCNNBlock(x, first_size, second_size, alpha, scope, data_format, reuse):
    with tf.variable_scope(scope, reuse=reuse) as vs:
        next_ = slim.conv2d(x, second_size, 3, 2, activation_fn=leaky_relu, weights_initializer=var_init(),
                            data_format=data_format)
        next_ = slim.conv2d(next_, second_size, 1, 1, activation_fn=leaky_relu, weights_initializer=var_init(),
                            data_format=data_format)*(1-alpha)
        x = slim.conv2d(x, first_size, 1, 1, activation_fn=leaky_relu, weights_initializer=var_init(),
                        data_format=data_format)
        x = slim.conv2d(x, first_size, 3, 1, activation_fn=leaky_relu, weights_initializer=var_init(),
                        data_format=data_format)
        x = slim.conv2d(x, second_size, 3, 1, activation_fn=leaky_relu, weights_initializer=var_init(),
                        data_format=data_format)
        x = slim.conv2d(x, second_size, 3, 2, activation_fn=leaky_relu, weights_initializer=var_init(),
                        data_format=data_format)
        # x = slim.avg_pool2d(x, 2, padding='VALID', data_format=data_format) 
        x = x*alpha + next_
        x = slim.conv2d(x, 3, 3, 1, activation_fn=None, data_format=data_format)
    variables = tf.contrib.framework.get_variables(vs)
    return x, variables


def DiscriminatorNSkipCNNLastBlock(x, hidden_num, scope, data_format, reuse):
    with tf.variable_scope(scope, reuse=reuse) as vs:
        x = slim.conv2d(x, hidden_num, 1, 1, activation_fn=leaky_relu, weights_initializer=var_init(),
                        data_format=data_format)
        x = minibatch_disc_concat(x)
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=leaky_relu, weights_initializer=var_init(),
                        data_format=data_format)
        x = slim.conv2d(x, hidden_num, 4, 1, padding='VALID', activation_fn=leaky_relu, weights_initializer=var_init(),
                        data_format=data_format)
        x = slim.fully_connected(x, 1, activation_fn=None)
    variables = tf.contrib.framework.get_variables(vs)
    return x, variables

def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

def get_conv_shape(tensor, data_format):
    shape = int_shape(tensor)
    # always return [N, H, W, C]
    if data_format == 'NCHW':
        return [shape[0], shape[2], shape[3], shape[1]]
    elif data_format == 'NHWC':
        return shape

def nchw_to_nhwc(x):
    return tf.transpose(x, [0, 2, 3, 1])

def nhwc_to_nchw(x):
    return tf.transpose(x, [0, 3, 1, 2])

def reshape(x, h, w, c, data_format):
    if data_format == 'NCHW':
        x = tf.reshape(x, [-1, c, h, w])
    else:
        x = tf.reshape(x, [-1, h, w, c])
    return x

def resize_nearest_neighbor(x, new_size, data_format):
    if data_format == 'NCHW':
        x = nchw_to_nhwc(x)
        x = tf.image.resize_nearest_neighbor(x, new_size)
        x = nhwc_to_nchw(x)
    else:
        x = tf.image.resize_nearest_neighbor(x, new_size)
    return x

def upscale(x, scale, data_format):
    _, h, w, _ = get_conv_shape(x, data_format)
    return resize_nearest_neighbor(x, (h*scale, w*scale), data_format)


def leaky_relu(features, name=None):
    out = tf.nn.relu(features, name)
    return tf.maximum(out, 0.2*out)


def batch_stdeps(x):
    return tf.sqrt(tf.reduce_mean(tf.square(x - tf.reduce_mean(x)), [1, 2, 3]) + 1e-8)


def minibatch_disc_concat(x):
    splt0, splt1 = tf.split(x, 2)
    slice_ = tf.expand_dims(splt0[:, 0, :, :], 1)
    shape = tf.shape(slice_)
    c0 = tf.fill(shape, tf.reduce_mean(batch_stdeps(splt0)))
    c1 = tf.fill(shape, tf.reduce_mean(batch_stdeps(splt1)))
    return tf.concat([x, tf.concat([c0, c1], 0)], 1)



    
    
    
