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

from errors import FirstInitialization
import numpy as np
import os
from subgraph import BuiltSubGraph
import tensorflow as tf
from utils import save_image


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


def init_subgraph(subgraph_name, type_, log_dir=None, convert_from=None,
                  log_dir_from=None, frozen=None):
    if convert_from is not None:
        with tf.Session(graph=tf.Graph()) as sess:
            subgraph = BuiltSubGraph(convert_from, type_, sess, log_dir_from)
            subgraph.save_clone(subgraph_name, log_dir, sess)
    try:
        with tf.Session(graph=tf.Graph()) as sess:
            subgraph = BuiltSubGraph(subgraph_name, type_, sess, log_dir)
            if frozen is not None:
                subgraph = subgraph.freeze(sess)
    except FirstInitialization:
        with tf.Session(graph=tf.Graph()) as sess:
            subgraph = BuiltSubGraph(subgraph_name, type_, sess, log_dir)
            if frozen is not None:
                subgraph = subgraph.freeze(sess)
    return subgraph


def strip_index(model_name):
    return '_'.join(model_name.split('_')[:-1])
