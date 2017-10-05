import os
from importlib import import_module
import numpy as np
import tensorflow as tf
from tqdm import trange
from utils import save_image


CONFIG_FILE = 'models.{}.config'
GRAPH_FILE = 'models.{}.graph'
MODEL_DIR = './models/'
LOGS_DIR = './logs/'
OUTPUTS = 'outputs'
INTERIM = 'outputs_interim'
LR = 'outputs_lr'
SUMMARY = 'summary'
STEP = 'step'


class Trainer(object):
    def __init__(self, model_name, model_type, config, log_folder, data_loader):
        self.config = config
        self.path = os.path.join(MODEL_DIR, model_name)
        self.log_dir = os.path.join(LOGS_DIR, log_folder)
        self.data_loader = data_loader

        self.max_step = config.max_step
        self.start_step = config.start_step
        self.save_step = config.save_step
        self.log_step = config.log_step
        self.lr_update_step = config.lr_update_step
        self.batch_size = config.batch_size
        self.z_num = config.z_num
        self.use_gpu = config.use_gpu

        config = import_module(CONFIG_FILE.format(model_name))
        graph = import_module(GRAPH_FILE.format(model_name))

        self.c_graph = graph.build_graph(config.config(model_type))
        self.output_fdict = dict([(var.name, var) for var in self.c_graph.graph.get_collection(OUTPUTS)])
        self.interim_fdict = dict([(var.name, var) for var in self.c_graph.graph.get_collection(INTERIM)])
        self.summary_name = self.c_graph.graph.get_collection(SUMMARY)[0] #should always be a single merge summary
        self.step = self.c_graph.graph.get_collection(STEP)[0] #should always be a single step variable
        
        with self.c_graph.graph.as_default():
            self.saver = tf.train.Saver()
            self.summary_writer = tf.summary.FileWriter(self.log_dir)

            print('Before SV init')
            sv = tf.train.Supervisor(logdir=self.log_dir,
                                     is_chief=True,
                                     saver=self.saver,
                                     summary_op=None,
                                     summary_writer=self.summary_writer,
                                     save_model_secs=1200,
                                     global_step=self.step,
                                     ready_for_local_init_op=None)
            print('After SV init')
            gpu_options = tf.GPUOptions(allow_growth=True)
            sess_config = tf.ConfigProto(allow_soft_placement=True,
                                         gpu_options=gpu_options)
            
            self.sess = sv.prepare_or_wait_for_session(config=sess_config)
            print('After sess init')


    def train(self):
        z_fixed = np.random.uniform(-1, 1, size=(self.batch_size, self.z_num))
        x_fixed = self.get_image_from_loader()
        save_image(x_fixed, '{}/x_fixed.png'.format(self.log_dir))

        for step in trange(self.start_step, self.max_step):

            feed_dict = self.c_graph.get_feed_dict(self.data_loader,
                                                   self.config,
                                                   self.sess)
            
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
                        out_str.append('{}: {.6f}'.format(var, val))
                print('[{}/{}]'.format(step, self.max_step) + ', '.join(out_str))

            #TODO: add subgraph component save
            #TODO: save output images

            if step % self.lr_update_step == self.lr_update_step - 1:
                self.sess.run(self.c_graph.graph.get_collection(LR))


    def get_image_from_loader(self):
        x = self.data_loader.eval(session=self.sess)
        if self.data_format == 'NCHW':
            x = x.transpose([0, 2, 3, 1])
        return x


