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
from networkx.algorithms.dag import ancestors, descendants
from importlib import import_module
from models.graphs.converter import convert
from models.model_utils import strip_index
from models.subgraph import SubGraph
from random import shuffle
import tensorflow as tf
from trainer import *
import trainer_config


CONFIG_FILE = 'models.configs.'
GRAPH_FILE = 'models.graphs.'
TRAINOPS_FILE = 'models.train_funcs.'
LOGS_DIR = './logs/'
BLCK = 'block'
DIR = 'dir'
DATA = 'data'
VARIABLES = '/variables'
INPT = '/input:0'
INPN = '/input{}:0'
OUTP = '/output:0'
OUTN = '/output{}:0'


class NetGen(object):


    def __init__(self, model_name, model_type, base_dataset, train_program,
                 timestamp=None, log_folder=None, branching=False, linked={}):
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
        self.linked = linked
        self.branching = branching

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
        base_block = self.G.graph['block_index']
        block_index = base_block
        for program in self.train_program:
            cur_log_dir = os.path.join(self.log_dir, program[DIR])
            if block_index == base_block:
                load_map = {}
                self.run_training(self.G, cur_log_dir, program[DATA], load_map, block_index)
            else:
                if block_index == last_index:
                    self.t_config.max_step = int(self.t_config.max_step*1.5)
                self.add_subgraphs(block_index)
                prev_log_dir = os.path.join(self.log_dir, self.train_program[block_index-1][DIR])
                load_map = self.build_load_map(prev_log_dir)
                if self.branching:
                    # split concat
                    
                    branches = []
                    for subgraph in self.G.nodes:
                        # if you're adding more than one branch sub_index needs to be scaled
                        try:
                            branch_type = self.G.nodes[subgraph]['branching']
                        except KeyError:
                            continue
                        branches.append((subgraph, branch_type, None))
                        try:
                            rsubgraph, rbranch_type = self.G.nodes[subgraph]['rev_pair']
                        except KeyError:
                            pass
                        else:
                            branches.append((rsubgraph, rbranch_type, subgraph)) # implicit ordering relation guarantees reverse follow forward

                    shuffle(branches) # need to prevent an ordering effect
                    for subgraph, branch_type, fsubgraph in branches:
                        # implement autobranch here
                        for i in range(self.G.graph['breadth_count'] - 1):
                            self.add_branch(branch_type, block_index, subgraph, fsubgraph, base_bridge, i)
                        
                        Gpart = self.get_partial_graph(subgraph) # needs to be implemented
                        self.run_training(Gpart, cur_log_dir, program[DATA], load_map, block_index)
                    
                else:
                    self.run_training(self.G, cur_log_dir, program[DATA], load_map, block_index)
            block_index += 1
            # after a training step finishes there should be no linked subgraphs
            self.linked = {}


    def get_partial_graph(self, subgraph): # needs to be implemented
        return None


    def build_load_map(self, log_dir):
        load_map = {}
        subgraphs = os.walk(log_dir).next()[1]
        for subgraph in subgraphs:
            load_map[subgraph] = os.path.join(log_dir, subgraph)
            try:
                linked_subs = self.linked[subgraph]
            except KeyError:
                pass
            else:
                for lsub in linked_subs:
                    load_map[lsub] = os.path.join(log_dir, subgraph)
        return load_map


    def convert_cg(self, G, load_map): #TODO need to add handling for branching, specifically reverse gen_pair concats/splits
        conngraph, inputs, outputs = convert(G, self.config, load_map)

        conngraph = self.t_ops.build_train_ops(conngraph, inputs, outputs, **G.graph)

        get_feed_dict = self.t_ops.build_feed_func(**G.graph)
        conngraph.attach_func(get_feed_dict)
        send_outputs = self.t_ops.build_send_func(**G.graph)
        conngraph.attach_func(send_outputs)

        return conngraph


    def run_training(self, G, log_folder, dataset, load_map, block_index):
        conngraph = self.convert_cg(G, load_map)
        conngraph.block_index = block_index
        trainer = Trainer(self.model_name, self.model_type, self.t_config,
                          log_folder, dataset, c_graph=conngraph)
        step = trainer.train()
        conngraph.save_subgraphs(trainer.log_dir, step, trainer.sess)
        del trainer


    def add_subgraph(self, new_subgraph, config, train, alphas, alpha_edge,
                     new_loss=None, train_set=None):
        if new_loss is None:
            self.G.graph['train_sets'][train].append(new_subgraph)
        else:
            self.G.graph['train_sets'][train][new_loss] = train_set
        self.G.graph['saver_pairs'].append(
            (new_subgraph, new_subgraph+VARIABLES)
        )
        self.G.add_nodes_from([
            (new_subgraph, {'config': config})
        ])
        self.G.add_edges_from([
            (alphas, new_subgraph, {'out': alpha_edge.format(block_index), 'in': alpha_edge.format(block_index)})
        ])


    def add_connection(self, new_subgraph, prev_subgraph, next_subgraph,
                       old_in, old_out, new_in, new_out, rev)
        self.G.add_edges_from([
            (prev_subgraph, new_subgraph, {'out': old_out, 'in': new_in}),
            (new_subgraph, next_subgraph, {'out': new_out, 'in': old_in}),
        ])
        if rev:
            self.G.edges[prev_subgraph, new_subgraph]['mod'] = prev_subgraph
        else:
            self.G.edges[new_subgraph, next_subgraph]['mod'] = next_subgraph

    def add_subgraphs(self, block_index):
        growth_types = self.G.graph['growth_types']
        alphas, alpha_edge = self.G.graph['alpha_tensor'].split('/')
        alpha_edge = '/'+alpha_edge
        for subgraph in growth_types:
            new_subgraph = growth_types[subgraph]
            nsub = self.get_new_name(new_subgraph)
            config = new_subgraph['config'].format(block_index).split('_')
            train = new_subgraph['train']
            self.add_subgraph(nsub, config, train, alphas, alpha_edge)

            rev = new_subgraph['rev']
            if rev:
                prev_subgraph = subgraph
                next_subgraph = list(self.G.successors(subgraph))[0]
            else:
                prev_subgraph = list(self.G.predecessors(subgraph))[0]
                next_subgraph = subgraph
            new_in = growth_types[subgraph]['in']
            new_out = growth_types[subgraph]['out']
            old_in = self.G.edges[prev_subgraph, next_subgraph]['in']
            old_out = self.G.edges[prev_subgraph, next_subgraph]['out']
            self.G.remove_edge(prev_subgraph, next_subgraph)
            self.add_connection(nsub, prev_subgraph, next_subgraph,
                                old_in, old_out, new_in, new_out, rev)


    def add_branch(self, branch_type, block_index, subgraph, forward_subgraph, base_bridge, branch_index):
        branch = self.G.graph['branch_types'][branch_type]
        rev = branch['rev']

        # add new bridge
        bridge = branch['bridge']
        nbridge = self.get_new_name(bridge)
        bridge_attrs = {'config': bridge['config']}
        try:
            inputs = bridge['inputs']
        except KeyError:
            pass
        else:
            bridge_attrs['inputs'] = inputs
        try:
            outputs = bridge['outputs']
        except KeyError:
            pass
        else:
            bridge_attrs['outputs'] = outputs

        # add new loss
        loss = branch['loss']
        nloss = self.get_new_name(loss)
        loss_attrs = {'outputs': loss['outputs']}

        self.G.add_nodes_from([
            (nbridge, bridge_attrs),
            (nloss, loss_attrs),
        ])

        # add concat
        if not rev:
            concat = branch['concat']
            old_concat = [dsub for dsub in descendants(self.G, subgraph) if self.G.graph['concat_type'] in dsub][0]
            self.split_nconcat(concat, old_concat, subgraph):

        # add load links
        try:
            linked_subs = self.linked[base_bridge]
        except KeyError:
            self.linked[base_bridge] = [nbridge]
        else:
            linked_subs.append(nbridge)

        # add graph
        alphas, alpha_edge = self.G.graph['alpha_tensor'].split('/')
        alpha_edge = '/'+alpha_edge
        new_subgraph = branch['new_subgraph']
        nsub = self.get_new_name(new_subgraph)
        config = new_subgraph['config'].format(block_index).split('_')
        
        self.add_subgraph(nsub, config, train, alphas, alpha_edge)

        # add connections
        if rev:
            prev_subgraph = nbridge
            next_subgraph = subgraph
        else:
            prev_subgraph = subgraph
            next_subgraph = nbridge
        old_in = bridge['in']
        old_out = new_subgraph['out'] # assumes that all branches are identical to their branch points which is reasonable
        new_in = new_subgraph['in']
        new_out = new_subgraph['out']
        self.add_connection(nsub, prev_subgraph, next_subgraph,
                            old_in, old_out, new_in, new_out, rev)

        edges = []
        if rev:
            fbridge = list(self.G.successors(forward_subgraph))[0]
            edges.append((fbridge, nloss, {'out': bridge['fout'], 'in': nloss['in']}))
        else:
            edges.append(
                (nbridge, nloss, {'out': bridge['out'][0], 'in': nloss['orig_in']}), # assumes eval_output is first index.. same below
                (nbridge, old_concat, {'out': bridge['out'][0], 'in': concat['in'].format(branch_index+2)}),
                (old_split, nloss, {'out': split['out'].format(branch_index+2), 'in': nloss['auto_in']}),
            ) # a_in needs to be attached with split


    def split_nsplit(self, split, old_split, branch_subgraph):
        successrs = self.G.successors(old_split)
        predecessr = list(self.G.predecessors(old_split))[0]
        lss_predecessrs = []
        for rsub in successrs:
            pred2 = self.G.predecessors(rsub) 
            lss_predecessrs.extend([(rsub, s) for s in pred2 if s != old_split]) # should dump the branch bridges attaching to the loss functions

        ancestor_sets = [(rsub, ancestors(s)) for rsub, s in lss_predecessrs]
        new_split_subs = [rsub for rsub, ancestrs in ancestor_sets if branch_subgraph not in ancestrs]

        for sub in new_split_subs:
            new_split = self.get_new_name(self.G.graph['split_type'])
            self.G.add_nodes_from([
                (new_split, {'config': ['count{}'.format(self.G.graph['breadth_count']+1)]})])
            new_in = split['in']
            new_out = split['out'].format(1) # always takes the index after real input (index 0)
            old_in = self.G.edges[old_split, sub]['in']
            old_out = self.G.edges[predecessr, old_split]['out']
            self.add_connection(new_concat, predecessr, sub,
                                old_in, old_out, new_in, new_out, False)


    def split_nconcat(self, concat, old_concat, branch_subgraph):
        predecessrs = self.G.predecessors(old_concat)
        successrs = list(self.G.successors(old_concat))
        
        ancestor_sets = [(rsub, ancestors(rsub)) for rsub in predecessrs]
        new_concat_subs = [rsub for rsub, ancestrs in ancestor_sets if branch_subgraph not in ancestrs]

        for sub in new_concat_subs:
            new_concat = self.get_new_name(concat['name'])
            self.G.add_nodes_from([
                (new_concat, {'config': ['count{}'.format(self.G.graph['breadth_count']+1)]})])
            new_in = concat['in'].format(1) # always takes the index after real input (index 0)
            new_out = concat['out']
            old_in = self.G.edges[old_concat, successrs[0]]['in']
            old_out = self.G.edges[sub, old_concat]['out']
            self.add_connection(new_concat, sub, successrs[0],
                                old_in, old_out, new_in, new_out, False)
            edges = []
            for suc in successrs[1:]:
                inpt = self.G.edges[old_concat, suc]['in']
                edges.append((new_concat, suc, {'out': new_out, 'in': inpt}))
            self.G.add_edges_from(edges)


    def check_ancestors_equal(self, ancestors):
        return reduce(lambda x, y: x|y, ancestrs) != reduce(lambda x, y: x&y, ancestrs)


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


    def get_new_name(self, subgraph_type):
        nsub = subgraph_type['name']
        name = strip_index(nsub)
        self.G.graph['counts'][name] += 1
        i = self.G.graph['counts'][name]
        nsub = nsub.format(i)
        return nsub
