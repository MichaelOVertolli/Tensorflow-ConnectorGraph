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

from copy import deepcopy
from data_loader import setup_sharddata
import glob
import networkx as nx
from networkx.algorithms.dag import ancestors, descendants, topological_sort
from networkx.exception import NetworkXError
import numpy as np
import os
from importlib import import_module
from models.errors import Error
from models.graphs.converter import convert
from models.model_utils import denorm_img_numpy, norm_img, save_image, strip_index
from models.subgraph import SubGraph
from random import shuffle
import re
from shutil import copytree
import tensorflow as tf


CONFIG_FILE = 'models.configs.'
GRAPH_FILE = 'models.graphs.'
TRAINOPS_FILE = 'models.train_funcs.'
LOG_DIR = './logs'
DATA_DIR = './data'
TEMP = './logs/temp/'
STEP = 'step'
NO_SHUFFLE = False
NORM_TRUE = True


class NetGenRunner(object):
    def __init__(self, model_name, model_type, load_dir,
                 graph_files, branching=True):

        self.load_dir = load_dir

        self.Gs = {}
        for g in graph_files:
            try:
                name, G = g
            except ValueError:
                name = g
                G = nx.read_gpickle(g)
            self.Gs[name.split('/')[-1]] = G

        self.load_map = self.build_load_map()

        config = import_module(CONFIG_FILE+model_name)
        self.t_ops = import_module(TRAINOPS_FILE+model_name)
        
        self.config = config.config(model_type)


    def run_losses(self, G, batch_size):
        concat = next(d for d in G.graph['data_inputs'] if G.graph['concat_type'] in d)
        losses =  G.graph['loss_tensors']['R']
        outputs = []
        tags = [next(s for s in G.predecessors(sub.split('/')[0])) for sub in G.graph['gen_outputs']['R']]
        pairs = []
        for i in range(1000000):
            try:
                feed_dict = self.c_graph.get_feed_dict(self)
            except tf.errors.OutOfRangeError:
                break
            else:
                raw_img = feed_dict[concat]
                if raw_img.shape[0] != batch_size:
                    shape = list(raw_img.shape)
                    shape[0] = batch_size - shape[0]
                    padding = np.zeros(shape)
                    raw_img = np.concatenate([raw_img, padding])
                for g_output in G.graph['gen_outputs']['G']:
                    r_input = G.graph['rev_inputs'][g_output]
                    feed_dict[r_input] = raw_img
                images = self.sess.run(G.graph['gen_outputs']['R'], feed_dict)
                # out = []
                # for imgs in images:
                    # feed_dict[concat] = imgs
                    # autoencoded = self.sess.run(self.sess.graph.get_tensor_by_name(G.graph['a_output']), feed_dict)
                try:
                    pad_cnt = shape[0]
                except (NameError, TypeError):
                    pass
                else:
                    cnt = batch_size - pad_cnt
                    raw_img = raw_img[:cnt]
                    images[0] = images[0][:cnt]
                    images[1] = images[1][:cnt]
                    shape = None
                outputs.append(np.stack([
                    np.mean(np.abs(images[0] - raw_img), axis=(1, 2, 3)),
                    np.mean(np.abs(images[1] - raw_img), axis=(1, 2, 3))]).T)
                # pairs.append([images[0], images[1], raw_img])
        return outputs, tags


    def subset_data(self, losses, keep_percent):
        losses = np.concatenate(losses)
        shape = losses.shape
        outputs = np.zeros(shape) == 1 # boolean init all False
        cnt = int(losses.shape[0] * keep_percent)
        while np.min(np.sum(outputs, 0)) < cnt:
            p = np.random.uniform(0, 1, shape)
            outputs = outputs | (p > losses) # > b/c similarity is 1 - loss
        return np.split(outputs, 2, axis=1)


    def generate(self, gen_input, gen_output, rev_inputs, loop_count):
        outputs = []
        for i in range(loop_count):
            z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_num))
            # feeds = [(gen_input, z)]
            # reverse = [(rev_input, np.zeros([self.batch_size, self.channels, self.img_size, self.img_size]))
            #            for rev_input in rev_inputs]
            # feeds = dict(feeds+reverse)
            feed_dict = self.c_graph.get_feed_dict(self)
            feed_dict[gen_input] = z
            output = self.sess.run(self.sess.graph.get_tensor_by_name(gen_output), feed_dict)
            outputs.append(output)
        return outputs


    def autoencode(self, a_output, data_inputs, inputs):
        outputs = []
        for inpt in inputs:
            feed_dict = self.c_graph.get_feed_dict(self)
            for d_inpt in data_inputs:
                feed_dict[d_inpt] = inpt
            output = self.sess.run(self.sess.graph.get_tensor_by_name(a_output), feed_dict)
            outputs.append(output)
        return outputs
    

    def reverse(self, rev_output, rev_input, inputs):
        outputs = []
        for inpt in inputs:
            feed_dict = self.c_graph.get_feed_dict(self)
            feed_dict[rev_input] = inpt
            output = self.sess.run(rev_output, feed_dict)
            outputs.append(output)
        return outputs
    

    def prep_sess(self, G, data_dir, fetch_size, resize, loop_count=0,
                  bool_mask=None, greyscale=False, data_format='NCHW'):
        # edges = [' '.join([fr, to, str(G.edges[fr, to])]) for fr, to in G.edges]
        # sorted(edges)
        # for e in edges:
        #     print e
        if greyscale:
            self.channels = 1
        else:
            self.channels = 3
        self.c_graph = self.convert_cg(G)
        self.step = self.c_graph.graph.get_collection(STEP)[0] #should always be a single step variable
        self.z_num = self.c_graph.config.z_num
        self.batch_size = self.c_graph.config.batch_size
        self.img_size = self.c_graph.config.img_size
        self.step = self.c_graph.graph.get_collection(STEP)[0] #should always be a single step variable

        data_path = os.path.join(DATA_DIR, data_dir)
        self.num_shards = len(glob.glob(data_path + '/*.tfrecords'))
        
        with self.c_graph.graph.as_default():
            with tf.device('/cpu:0'):
                self.data, self.data_loader, self.init = setup_sharddata(data_path,
                                                                         fetch_size,
                                                                         self.batch_size,
                                                                         loop_count,
                                                                         greyscale,
                                                                         NORM_TRUE,
                                                                         NO_SHUFFLE,
                                                                         bool_mask,
                                                                         resize,
                                                                         data_format)

            self.sv = tf.train.Supervisor(logdir=None,
                                          is_chief=True,
                                          local_init_op=self.init,
                                          saver=None,
                                          summary_writer=None,
                                          save_model_secs=0,
                                          global_step=None,
                                          init_fn=self.c_graph.initialize)

            gpu_options = tf.GPUOptions(allow_growth=True)
            sess_config = tf.ConfigProto(allow_soft_placement=True,
                                         gpu_options=gpu_options)
            
            self.sess = self.sv.prepare_or_wait_for_session(config=sess_config)


    def build_load_map(self):
        load_map = {}
        subgraphs = [s for s in os.walk(self.load_dir).next()[1]]
        for subgraph in subgraphs:
            load_map[subgraph] = [
                os.path.join(self.load_dir, subgraph),
                None,
                None,
                None,
            ]
        return load_map


    def setup_temp(self):
        if not os.path.exists(TEMP):
            os.makedirs(TEMP)
        files = os.listdir(TEMP)
        for f in files:
            path = os.path.join(TEMP, f)
            if os.path.isfile(path):
                os.unlink(path)


    def convert_cg(self, G):
        conngraph, inputs, outputs = convert(G, self.config, self.load_map)

        self.setup_temp()
        conngraph = self.t_ops.build_train_ops(LOG_DIR, conngraph, inputs, outputs, **G.graph)

        get_feed_dict = self.t_ops.build_feed_func(**G.graph)
        conngraph.attach_func(get_feed_dict)

        return conngraph


    def freeze_subgraph(self, G, subgraph, log_folder, branching):
        graph = G.graph
        if subgraph in graph['gen_input']:
            graph['gen_input'] = '/'.join([subgraph, graph['gen_input']])
        attributes = {'frozen': True}
        try:
            outputs = G.nodes[subgraph]['outputs']
        except KeyError:
            pass
        else:
            try:
                outputs = '/'.join(['', subgraph+outputs])
            except TypeError:
                outputs = ['/'.join(['', subgraph+o]) for o in outputs]
            attributes['outputs'] = outputs
        try:
            inputs = G.nodes[subgraph]['inputs']
        except KeyError:
            pass
        else:
            try:
                inputs = '/'.join(['', subgraph+inputs])
            except TypeError:
                inputs = ['/'.join(['', subgraph+i]) for i in inputs]
            attributes['inputs'] = inputs
        G.add_nodes_from([
            (subgraph, attributes),
        ])
        for psub in G.predecessors(subgraph):
            in_ = G.edges[psub, subgraph]['in']
            outpts = G.edges[psub, subgraph]['out']
            try:
                inpts = '/'.join(['', subgraph+in_])
            except TypeError:
                inpts = ['/'.join(['', subgraph+i]) for i in in_]
            G.add_edges_from([
                (psub, subgraph, {'out': outpts, 'in': inpts})
            ])
        for ssub in G.successors(subgraph):
            inpts = G.edges[subgraph, ssub]['in']
            out_ = G.edges[subgraph, ssub]['out']
            try:
                outpts = '/'.join(['', subgraph+out_])
            except TypeError:
                outpts = ['/'.join(['', subgraph+o]) for o in out_]
            G.add_edges_from([
                (subgraph, ssub, {'out': outpts, 'in': inpts})
            ])


    def save_img(self, img, name, data_format='NCHW'):
        """Saves a batch of images.

        Arguments:
        img         := (numpy array) a batch of images
        name        := (str) the name for the image file to be saved in log_folder
        data_format := (str) the format for the input data 

        """
        save_image(denorm_img_numpy(img, 'NCHW'), os.path.join(LOG_DIR, name))


    def close_sess(self):
        self.sv.stop()
        self.sv.wait_for_stop()
        self.sess.close()


    def close(self):
        del self.load_dir

        del self.channels

        self.step = None
        self.z_num = None
        self.batch_size = None
        self.img_size = None

        self.close_sess()
        del self.sess
        del self.sv

        del self.num_shards

        del self.init
        del self.data_loader
        del self.data
        
        self.c_graph.close()
        del self.c_graph

        self.config = None

        for k in self.Gs:
            self.Gs[k].clear()
        self.Gs.clear()
        del self.Gs
        self.load_map.clear()
        del self.load_map

        del self.t_ops
