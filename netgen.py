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

import networkx as nx
from importlib import import_module
from models.graphs.converter import convert
from models.subgraph import SubGraph
import tensorflow as tf
from trainer import *
import trainer_config


CONFIG_FILE = 'models.configs.'
GRAPH_FILE = 'models.graphs.'
TRAINOPS_FILE = 'models.train_funcs.'
LOGS_DIR = './logs/'
BASE = 'base_block'
BLCK = 'block'
DIR = 'dir'
DATA = 'data'
VARIABLES = '/variables'


class NetGen(object):


    def __init__(self, model_name, model_type, base_dataset, train_program,
                 timestamp=None, log_folder=None):
        if timestamp is None:
            self.timestamp = get_time()
        else:
            self.timestamp = timestamp
        if log_folder is None:        
            log_folder = '_'.join(['NETGEN', base_dataset, model_name, model_type, self.timestamp])
        self.log_dir = os.path.join(LOGS_DIR, log_folder)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            for program in train_program:
                os.makedirs(os.path.join(self.log_dir, program[DIR]))

        config = import_module(CONFIG_FILE+model_name)
        graph = import_module(GRAPH_FILE+model_name)
        self.t_ops = import_module(TRAINOPS_FILE+model_name)

        self.config = config.config(model_type)
        self.G = graph.build(self.config)
        
        self.model_name = model_name
        self.model_type = model_type
        self.train_program = train_program
        self.t_config = trainer_config.config()
        self.t_config.max_step = self.config.alpha_update_steps*2


    def run(self):
        last_index = len(self.train_program)-1
        for block_index, program in enumerate(self.train_program):            
            if block_index == 0:
                load_map = {}
            else:
                if block_index == last_index:
                    self.t_config.max_step = int(self.t_config.max_step*1.5)
                self.add_subgraphs(block_index)
                prev_log_dir = os.path.join(self.log_dir, self.train_program[block_index-1][DIR])
                load_map = self.build_load_map(prev_log_dir)
            cur_log_dir = os.path.join(self.log_dir, program[DIR])
            self.run_training(cur_log_dir, program[DATA], load_map, block_index)
            


    def build_load_map(self, log_dir):
        load_map = {}
        subgraphs = os.walk(log_dir).next()[1]
        for subgraph in subgraphs:
            load_map[subgraph] = os.path.join(log_dir, subgraph)
        return load_map


    def convert_cg(self, load_map):
        conngraph, inputs, outputs = convert(self.G, self.config, load_map)

        conngraph = self.t_ops.build_train_ops(conngraph, inputs, outputs, **self.G.graph)

        get_feed_dict = self.t_ops.build_feed_func(**self.G.graph)
        conngraph.attach_func(get_feed_dict)
        send_outputs = self.t_ops.build_send_func(**self.G.graph)
        conngraph.attach_func(send_outputs)

        return conngraph


    def run_training(self, log_folder, dataset, load_map, block_index):
        conngraph = self.convert_cg(load_map)
        conngraph.block_index = block_index
        trainer = Trainer(self.model_name, self.model_type, self.t_config,
                          log_folder, dataset, c_graph=conngraph)
        step = trainer.train()
        conngraph.save_subgraphs(trainer.log_dir, step, trainer.sess)
        del trainer


    def add_subgraphs(self, block_index):
        new_subgraphs = []
        growth_types = self.G.graph['growth_types']
        alphas, alpha_edge = self.G.graph['alpha_tensor'].split('/')
        alpha_edge = '/'+alpha_edge
        for subgraph in growth_types:
            new_subgraph = growth_types[subgraph]['new_subgraph'].format(block_index)
            config = growth_types[subgraph]['config'].format(block_index).split('_')
            train = growth_types[subgraph]['train']
            self.G.graph['train_sets'][train].append(new_subgraph)
            self.G.graph['saver_pairs'].append(
                (new_subgraph, new_subgraph+VARIABLES)
            )
            new_subgraphs.append((new_subgraph, {'config': config}))
            self.G.add_edges_from([
                (alphas, new_subgraph, {'out': alpha_edge.format(block_index), 'in': alpha_edge.format(block_index)})
            ])
        self.G.add_nodes_from(new_subgraphs)
        edges = [_ for _ in self.G.edges]
        for fr, to in edges:
            try:
                subgraph = self.G.edges[fr, to]['mod']
            except KeyError:
                continue
            else:
                new_subgraph = growth_types[subgraph]['new_subgraph'].format(block_index)
                new_in = growth_types[subgraph]['in']
                new_out = growth_types[subgraph]['out']
                old_in = self.G.edges[fr, to]['in']
                old_out = self.G.edges[fr, to]['out']
                self.G.remove_edge(fr, to)
                self.G.add_edges_from([
                    (fr, new_subgraph, {'in': old_in, 'out': new_out}),
                    (new_subgraph, to, {'in': new_in, 'out': old_out})
                ])
                if subgraph == fr:
                    self.G.edges[fr, new_subgraph]['mod'] = subgraph
                elif subgraph == to:
                    self.G.edges[new_subgraph, to]['mod'] = subgraph
                else:
                    pass # throw error


    def save_graph(self):
        nx.write_gpickle(self.G, os.path.join(self.log_dir, 'graph_output_{}.pkl'.format(get_time())))


    def load_graph(self, timestamp):
        self.G = nx.read_gpickle(os.path.join(self.log_dir, 'graph_output_{}.pkl'.format(timestamp)))


    def freeze(self, program, with_train_ops=False):
        prev_log_dir = os.path.join(self.log_dir, program[DIR])
        load_map = self.build_load_map(prev_log_dir)
        conngraph, inputs, outputs = convert(self.G, self.config, load_map)
        if with_train_ops:
            conngraph = self.t_ops.build_train_ops(conngraph, inputs, outputs, **self.G.graph)
        else:
            with tf.Session(graph=tf.Graph()) as sess:
                conngraph.connect_graph(inputs, outputs, sess)
                subgraph = SubGraph(self.model_name, self.config.name, conngraph.graph)
                freeze = subgraph.freeze(sess)
        return freeze
