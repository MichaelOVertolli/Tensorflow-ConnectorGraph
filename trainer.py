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
from data_loader import setup_sharddata
import os
from importlib import import_module
from models.errors import NANError, ModalCollapseError
from models.model_utils import denorm_img_numpy
import numpy as np
import tensorflow as tf
from tqdm import trange
from utils import save_image


CONFIG_FILE = 'models.{}.config'
GRAPH_FILE = 'models.{}.graph'
MODEL_DIR = './models/'
LOGS_DIR = './logs/'
DATA_DIR = './data/'
OUTPUTS = 'outputs'
INTERIM = 'outputs_interim'
LR = 'outputs_lr'
SUMMARY = 'summary'
STEP = 'step'
REPEAT_INF = -1
NORM_TRUE = True
SHUFFLE_TRUE = True


class Trainer(object):
    def __init__(self, model_name, model_type, config,
                 data_name, fetch_size, resize,
                 bool_mask=None, log_folder=None, c_graph=None,
                 save=True, greyscale=False):
        self.config = config
        self.path = os.path.join(MODEL_DIR, model_name)
        if log_folder is None:
            log_folder = '_'.join([data_name, model_name, model_type, get_time()])
        if LOGS_DIR not in log_folder:
            self.log_dir = os.path.join(LOGS_DIR, log_folder)
        else:
            self.log_dir = log_folder
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.data_dir = os.path.join(DATA_DIR, data_name)
        self.greyscale = greyscale
        self.fetch_size = fetch_size
        self.resize = resize
        self.bool_mask = bool_mask

        self.max_step = config.max_step
        self.start_step = config.start_step
        self.save_step = config.save_step
        self.log_step = config.log_step
        self.lr_update_step = config.lr_update_step
        
        self.data_format = config.data_format
        self.use_gpu = config.use_gpu
        # self.run_type = run_type

        if c_graph is None:
            config = import_module(CONFIG_FILE.format(model_name))
            graph = import_module(GRAPH_FILE.format(model_name))

            self.c_graph = graph.build_graph(config.config(model_type))
        else:
            self.c_graph = c_graph
        self.output_fdict = dict([(var.name, var) for var in self.c_graph.graph.get_collection(OUTPUTS)])
        self.interim_fdict = dict([(var.name, var) for var in self.c_graph.graph.get_collection(INTERIM)])
        self.summary_name = self.c_graph.graph.get_collection(SUMMARY)[0].name #should always be a single merge summary
        self.step = self.c_graph.graph.get_collection(STEP)[0] #should always be a single step variable

        self.z_num = self.c_graph.config.z_num
        self.batch_size = self.c_graph.config.batch_size
        self.img_size = self.c_graph.config.img_size
        
        with self.c_graph.graph.as_default():
            with tf.device('/cpu:0'):
                self.data, self.data_loader, self.init = setup_sharddata(self.data_dir,
                                                                         self.fetch_size,
                                                                         self.batch_size,
                                                                         REPEAT_INF,
                                                                         self.greyscale,
                                                                         NORM_TRUE,
                                                                         SHUFFLE_TRUE,
                                                                         self.bool_mask,
                                                                         self.resize,
                                                                         self.data_format)
            # self.data_loader = get_loader(self.data_dir,
            #                               self.batch_size,
            #                               self.img_size,
            #                               self.data_format,
            #                               self.run_type,
            #                               self.greyscale)
            if save:
                self.saver = tf.train.Saver()
            else:
                self.saver = None
            self.summary_writer = tf.summary.FileWriter(self.log_dir)

            
            self.sv = tf.train.Supervisor(logdir=self.log_dir,
                                          is_chief=True,
                                          saver=self.saver,
                                          local_init_op=self.init,
                                          summary_op=None,
                                          summary_writer=self.summary_writer,
                                          save_model_secs=1200,
                                          global_step=self.step,
                                          init_fn=self.c_graph.initialize)
            
            gpu_options = tf.GPUOptions(allow_growth=True)
            sess_config = tf.ConfigProto(allow_soft_placement=True,
                                         gpu_options=gpu_options)
            
            self.sess = self.sv.prepare_or_wait_for_session(config=sess_config)

    def train(self):
        for step in trange(self.start_step, self.max_step):

            feed_dict = self.c_graph.get_feed_dict(self)
            
            fetch_dict = dict([_ for _ in self.output_fdict.items()])

            if step % self.log_step == 0:
                fetch_dict.update(dict([_ for _ in self.interim_fdict.items()]))

            result = self.sess.run(fetch_dict, feed_dict)

            if step % self.log_step == 0:
                self.summary_writer.add_summary(result[self.summary_name], step)
                self.summary_writer.flush()

                out_str = []
                for var, val in result.items():
                    if var == self.summary_name:
                        continue
                    else:
                        out_str.append('{}: {:.6f}'.format(var, val))
                str_ = '[{}/{}]\n'.format(step, self.max_step) + '\n'.join(out_str)
                print(str_)
                if ' nan' in str_:
                    raise NANError()
                elif 'k_t:0: 1' in str_:
                    raise ModalCollapseError()

            if (step+1) % (self.log_step * 10) == 0:
                self.c_graph.send_outputs(self, step)

            if step % self.lr_update_step == self.lr_update_step - 1:
                self.sess.run(self.c_graph.graph.get_collection(LR))
        return step


    def get_image_from_loader(self):
        x = self.data_loader.eval(session=self.sess)
        # if self.data_format == 'NCHW':
        #     x = x.transpose([0, 2, 3, 1])
        return x


    def close_sess(self):
        self.sv.stop()
        self.sv.wait_for_stop()
        self.sess.close()


    def close(self):
        self.close_sess()
        
        self.config = None # handled by netgen
        self.bool_mask = None # handled by netgen
        
        del self.path
        del self.log_dir
        del self.data_dir
        del self.greyscale
        del self.fetch_size
        del self.resize

        del self.max_step
        del self.start_step
        del self.save_step
        del self.log_step
        del self.lr_update_step
        
        del self.data_format
        del self.use_gpu

        self.c_graph.close()
        del self.c_graph

        self.output_fdict.clear()
        del self.output_fdict
        self.interim_fdict.clear()
        del self.interim_fdict
        del self.summary_name
        del self.step

        self.z_num = None
        self.batch_size = None
        self.img_size = None
        
        del self.data_loader
        del self.data
        del self.init

        del self.saver
        del self.summary_writer

            
        del self.sv
            
        del self.sess


def get_time():
    return datetime.now().strftime("%m%d_%H%M%S")

