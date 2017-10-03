import os
import tensorflow as tf
from importlib import import_module
from models.cqs_cg.graph import build_graph
from models.cqs_cg.config import config

CONFIG_FILE = 'models.{}.config'
GRAPH_FILE = 'models.{}.graph'
MODEL_DIR = './models/'
LOGS_DIR = './logs/'

class Trainer(object):
    def __init__(self, model_name, model_type, log_folder, data_loader):
        self.path = os.path.join(MODEL_DIR, model_name)
        self.log_dir = os.path.join(LOGS_DIR, log_folder)
        self.data_loader = data_loader

        config = import_module(CONFIG_FILE.format(model_name))
        graph = import_module(GRAPH_FILE.format(model_name))

        self.c_graph = graph.build_graph(config.config(model_type))
        
        with self.c_graph.graph.as_default():
            self.saver = tf.train.Saver()
            self.summary_writer = tf.summary.FileWriter(self.log_dir)

        sv = tf.train.Supervisor(logdir=self.log_dir,
                                 is_chief=True,
                                 saver=self.saver,
                                 summary_op=None,
                                 summary_writer=self.summary_writer,
                                 save_model_secs=1200,
                                 global_step=self.step,
                                 ready_for_local_init_op=None)

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=gpu_options)

        self.sess = sv.prepare_or_wait_for_session(config=sess_config)
