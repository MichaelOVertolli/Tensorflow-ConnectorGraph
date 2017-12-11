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

from datetime import datetime
from data_loader import get_loader
from importlib import import_module
import os
import numpy as np
import tensorflow as tf
from tqdm import trange
from models.model_utils import save_image, denorm_img_numpy, norm_img


LOGS_DIR = './logs/'
DATA_DIR = './data/'


class Runner(object):
    def __init__(self, frozen_graphs, log_folder, data_name='CelebA',
                 batch_size=16, z_num=128, img_size=64, data_format='NCHW'):
        """Initialize Tensorflow backend for pre-trained GAN FrozenGraph runs

        Arguments:
        frozen_graphs := (list of FrozenGraphs) FrozenGraph objects that will
                         be loaded into the graph to be run
        log_folder    := (str) logging folder in ./logs 
        data_name     := (str) data set folder in ./data for loading real data
        batch_size    := (int) size of the batch for processing
        z_num         := (int) size of the random input vector for GANs
        img_size      := (int) size of the input images
        data_format   := (str) format of the data (GPU is NCHW)

        Public functions:
        get_image_from_loader()
        save_image(img, name, data_format)

        """
        self.log_dir = os.path.join(LOGS_DIR, log_folder)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.data_dir = os.path.join(DATA_DIR, data_name)
        self.run_type = 'train'
        self.data_format = data_format

        self.frozen_graphs = frozen_graphs

        self.z_num = z_num
        self.batch_size = batch_size
        self.img_size = img_size            
        
        with tf.Graph().as_default():
            for fgraph in self.frozen_graphs:
                fgraph.restore(self.frozen_graph.name,
                               self.frozen_graph.config_type,
                               None)
            self.data_loader = get_loader(self.data_dir,
                                          self.batch_size,
                                          self.img_size,
                                          self.data_format,
                                          self.run_type)

            sv = tf.train.Supervisor(logdir=None,
                                     is_chief=True,
                                     saver=None,
                                     summary_op=None,
                                     summary_writer=None,
                                     save_model_secs=0,
                                     global_step=None,
                                     ready_for_local_init_op=None)
            
            gpu_options = tf.GPUOptions(allow_growth=True)
            sess_config = tf.ConfigProto(allow_soft_placement=True,
                                         gpu_options=gpu_options)
            
            self.sess = sv.prepare_or_wait_for_session(config=sess_config)


    def get_image_from_loader(self):
        """Pulls data from the dataset."""
        x = self.data_loader.eval(session=self.sess)
        return norm_img(x)


    def save_img(self, img, name, data_format='NCHW'):
        """Saves a batch of images.

        Arguments:
        img         := (numpy array) a batch of images
        name        := (str) the name for the image file to be saved in log_folder
        data_format := (str) the format for the input data 

        """
        save_image(denorm_img_numpy(img, 'NCHW'), os.path.join(self.log_dir, name))


def get_time():
    """Returns a time string for folder uniqueness."""
    return datetime.now().strftime("%m%d_%H%M%S")
