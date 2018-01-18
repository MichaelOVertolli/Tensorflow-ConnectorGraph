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

from models.errors import TessellaterError
from models.model_utils import denorm_img_numpy, save_image
from models.subgraph import BuiltSubGraph
import numpy as np
import os
from PIL import Image
from random import randint
from runner import Runner
import tensorflow as tf


# t = Tessellater('./logs/grass_res_cg_ebm_full_reverse_scaled_began_gmsm_b16_z128_sz32_h128_g0.7_elu_pconv_wxav_0106_161826/res_generator_loop_0', 'grass', 'res_generator_loop_0', 'z128_sz32_h128_elu_pconv_wxav', 32, 128, [256, 256], 'gen_grass.jpg', 'testing/tes_grass/')


class Tessellater(object):

    def __init__(self, model_dir, data_name, model_name, config_type,
                 input_isize, input_zsize, output_size, file_name, log_folder,
                 inputs=None, frozen_graph=None):
        if frozen_graph is None and model_dir is None:
            raise TessellaterError('model_dir cannot be None. Specify a valid model path.')
        self.output_size = output_size # assumes output_size is [w, h]
        self.file_name = file_name
        self.input_isize = input_isize
        self.input_zsize = input_zsize
        self.log_folder = log_folder
        if inputs is not None:
            self.inputs = inputs
        if frozen_graph is None:
            with tf.Session(graph=tf.Graph()) as session:
                self.graph = BuiltSubGraph(model_name, config_type, session, model_dir)
                self.graph = self.graph.freeze(session)
        else:
            self.graph = frozen_graph
        self.runner = Runner([self.graph], log_folder, data_name, img_size=input_isize)


    def fill_base(self, type_):
        if type_ == 'random':
            self.img = np.random.uniform(-1.0, 1.0, [1, 3, self.output_size[1], self.output_size[0]]) #NCHW
        elif type_ == 'gen_fill':
            w, h = self.output_size
            isize = self.input_isize
            w = int(np.ceil(w/float(isize)))
            h = int(np.ceil(h/float(isize)))
            n = h*w
            self.img = np.zeros([1, 3, h*isize, w*isize])
            if self.inputs is not None:
                inputs = dict([_ for _ in self.inputs.items()])
            else:
                inputs = {}
            inputs[self.zinput] = np.random.uniform(-1, 1, [n, self.input_zsize])
            parts = self.runner.sess.run(self.zoutput, inputs)
            for i in range(h):
                for j in range(w):
                    self.img[0, :, i*isize:(i+1)*isize, j*isize:(j+1)*isize] = parts[i*w+j, :, :, :]
            self.img = self.img[:, :, :self.output_size[1], :self.output_size[0]]
        else:
            pass #throw error


    def tessellate(self, iters):
        w, h = self.output_size
        isize = self.input_isize
        w -= (isize+1)
        h -= (isize+1)
        for i in range(iters):
            w_ = randint(0, w)
            h_ = randint(0, h)
            if self.inputs is not None:
                inputs = dict([_ for _ in self.inputs.items()])
            else:
                inputs = {}
            inputs[self.rinput] = self.img[:, :, h_:h_+isize, w_:w_+isize]
            m_img = self.runner.sess.run(self.routput, inputs)
            self.img[:, :, h_:h_+isize, w_:w_+isize] = m_img


    def tessellate2(self, iters):
        w, h = self.output_size
        isize = self.input_isize
        w -= (isize+1)
        h -= (isize+1)
        ws = np.random.randint(0, w, iters)
        hs = np.random.randint(0, h, iters)
        imgs = np.zeros([iters, 3, isize, isize])
        for i in range(iters):
            h_, w_ = hs[i], ws[i]
            imgs[i, :, :, :] = self.img[:, :, h_:h_+isize, w_:w_+isize]
        if self.inputs is not None:
            inputs = dict([_ for _ in self.inputs.items()])
        else:
            inputs = {}
        inputs[self.rinput] = imgs
        m_imgs = self.runner.sess.run(self.routput, inputs)
        for i in range(iters):
            h_, w_ = hs[i], ws[i]
            self.img[:, :, h_:h_+isize, w_:w_+isize] = m_imgs[i, :, :, :]


    def save_img(self, mod):
        save_image(denorm_img_numpy(self.img, 'NCHW'),
                   os.path.join('./logs/', self.log_folder, self.file_name.format(mod)))


    def set_rinput(self, rinput):
        self.rinput = rinput


    def set_zinput(self, zinput):
        self.zinput = zinput


    def set_routput(self, routput):
        self.routput = routput


    def set_zoutput(self, zoutput):
        self.zoutput = zoutput
