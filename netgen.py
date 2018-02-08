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
from networkx.algorithms.dag import ancestors, descendants, topological_sort
from networkx.exception import NetworkXError
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
                            branches.append((rsubgraph, rbranch_type, subgraph)) # implicit ordering relation guarantees reverse follows forward

                    shuffle(branches) # need to prevent an ordering effects
                    for subgraph, branch_type, fsubgraph in branches:
                        # implement autobranch here
                        # bridge_name = strip_index(self.G.graph['branch_types'][branch_type]['bridge']['name'])
                        if rev:
                            ds = topological_sort(nx.subgraph(descendants(self.G, subgraph)))
                            base_train_set = []
                            for d in ds:
                                if branch_type in d:
                                    base_train_set.append(d)
                                else:
                                    break
                            base_bridge = next(n.G.successors(next(n.G.successors(subgraph))))
                        else:
                            base_train_set = list(ancestors(self.G, subgraph))
                            base_bridge = next(self.G.successors(next(self.G.successors(subgraph))))
                        # base_bridge = next(sub for sub in base_train_set if bridge_name in sub)
                        base_train_set.append(subgraph)
                        base_train_set = [sub for sub in base_train_set if branch_type in sub]
                        losses = [self.G.graph['loss_tensors']['D'].split('/')[0],
                                  self.G.graph['loss_tensors']['U'][0].split('/')[0]]
                        losses.extend([s for s in self.G.successors(base_bridge) if strip_index(losses[0]) in s])
                        for i in range(self.G.graph['breadth_count'] - 1):
                            nloss = self.add_branch(branch_type, block_index, subgraph, fsubgraph,
                                                    base_bridge, base_train_set, i)
                            losses.append(nloss)
                        if self.G.reversable:
                            if fsubgraph is not None:
                                self.get_partial_graph(fsubgraph, subgraph, losses) 
                        else:
                            Gpart = self.get_partial_graph(subgraph, None, losses)
                        self.run_training(Gpart, cur_log_dir, program[DATA], load_map, block_index)
                    
                else:
                    self.run_training(self.G, cur_log_dir, program[DATA], load_map, block_index)
            block_index += 1
            # after a training step finishes there should be no linked subgraphs
            self.linked = {}


    def get_partial_graph(self, fsubgraph, rsubgraph, losses):
        # creating partial graph
        base = [fsubgraph, self.G.graph['alpha_tensor'].split('/')[0]]
        base.extend(losses)
        try:
            top = set(descendants(self.G, rsubgraph))
        except NetworkXError:
            top = set(ancestors(self.G, fsubgraph))
        else:
            base.append(rsubgraph)
            top &= set(ancestors(self.G, fsubgraph))
        descendnts = set(descendants(self.G, fsubgraph))
        bot = descendnts & set(ancestors(self.G, losses[-1])) # not Disc or mad loss
        full = list(top | bot)+base
        partial_graph = nx.subgraph(self.G, full).copy()

        # adjusting graph attributes
        graph = partial_graph.graph
        saver_pairs = {}
        for sub in partial_graph.nodes:
            try:
                variables = graph['saver_pairs'][sub]
            except KeyError:
                continue
            else:
                saver_pairs[sub] = variables
        graph['saver_pairs'] = saver_pairs.items()
        
        img_pairs = []
        gen_outputs = []
        for i, sub in enumerate(self.G.successors(fsubgraph)):
            bridge = next(self.G.successors(sub))
            bridge_out = bridge+graph['img_pairs']['G'][1]
            img_pairs.append(('G{}'.format(i), bridge_out))
            gen_outputs.append(bridge_out)
            if self.G.graph['reversable']:
                img_pairs.append(('R{}'.format(i), bridge+graph['img_pairs']['R'][1]))
        for sub in self.G.predecessors(losses[-1]): # not Disc or mad loss
            if graph['img_pairs']['A_'][0] in sub:
                for i in range(1, self.G.graph['breadth_count']+1):
                    img_pairs.append(('A_G{}'.format(i), sub+graph['img_pairs']['A_'][1].format(i)))
                img_pairs.append(('A_D', sub+graph['img_pairs']['A_'][1].format(0)))
                break # there's only one relevant subgraph
        graph['img_pairs'] = img_pairs
        graph['gen_outputs']['G'] = gen_outputs

        data_inputs = []
        data_inputs.append(graph['data_inputs']['loss'])
        for sub in descendants(self.G, fsubgraph):
            if graph['concat_type'] in sub:
                data_inputs.append(sub+graph['data_inputs']['concat'])
                break
        graph['data_inputs'] = data_inputs
        
        return partial_graph


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
                     block_index, loss, train_set=None):
        if train_set is None:
            t_set = self.G.graph['train_sets'][train]
            try:
                t_set[loss].append(new_subgraph)
            except TypeError:
                t_set.append(new_subgraph)
        else:
            self.G.graph['train_sets'][train][loss] = train_set
        # saver_pairs.append((new_subgraph, new_subgraph+VARIABLES))
        # self.G.graph['saver_pairs'].extend(saver_pairs)
        self.G.graph['saver_pairs'][new_subgraph] = new_subgraph+VARIABLES
        self.G.add_nodes_from([
            (new_subgraph, {'config': config})
        ])
        self.G.add_edges_from([
            (alphas, new_subgraph, {'out': alpha_edge.format(block_index), 'in': alpha_edge.format(block_index)})
        ])


    def add_connection(self, new_subgraph, prev_subgraph, next_subgraph,
                       old_in, old_out, new_in, new_out, rev, nomod=False):
        self.G.add_edges_from([
            (prev_subgraph, new_subgraph, {'out': old_out, 'in': new_in}),
            (new_subgraph, next_subgraph, {'out': new_out, 'in': old_in}),
        ])
        growth_types = self.G.graph['growth_types']
        if prev_subgraph in growth_types or next_subgraph in growth_types:
            prev = prev_subgraph
            nxt = next_subgraph
        else:
            base_name = strip_index(new_subgraph)
            prev = base_name
            nxt = base_name
        if not nomod:
            if rev:
                self.G.edges[prev_subgraph, new_subgraph]['mod'] = prev
            else:
                self.G.edges[new_subgraph, next_subgraph]['mod'] = nxt

    def add_subgraphs(self, block_index):
        growth_types = self.G.graph['growth_types']
        branch_types = self.G.graph['branch_types']
        alphas, alpha_edge = self.G.graph['alpha_tensor'].split('/')
        alpha_edge = '/'+alpha_edge
        subgraphs = []
        for prev_subgraph, next_subgraph in self.G.edges:
            try:
                key = self.G.edges[prev_subgraph, next_subgraph]['mod']
            except KeyError:
                continue
            
            try:
                new_subgraph = growth_types[key]
            except KeyError:
                new_subgraph = branch_types[key]['new_subgraph']
            nsub = self.get_new_name(new_subgraph)
            config = new_subgraph['config'].format(block_index).split('_')
            train = new_subgraph['train']

            rev = new_subgraph['rev']
            if rev:
                for sub in self.G.nodes:
                    try:
                        rev_pair = self.G.nodes[sub]
                    except KeyError:
                        continue
                    if rev_pair == nsub:
                        loss = sub
            else:
                loss = next((s for s in self.G.successors(next_subgraph) if 'loss' in s), None)
                
            self.add_subgraph(nsub, config, train, alphas, alpha_edge, block_index, loss)
            new_in = new_subgraph['in']
            new_out = new_subgraph['out']
            old_in = self.G.edges[prev_subgraph, next_subgraph]['in']
            old_out = self.G.edges[prev_subgraph, next_subgraph]['out']
            self.G.remove_edge(prev_subgraph, next_subgraph)
            self.add_connection(nsub, prev_subgraph, next_subgraph,
                                old_in, old_out, new_in, new_out, rev)


    def add_branch(self, branch_type, block_index, subgraph, forward_subgraph, base_bridge,
                   base_train_set, branch_index):
        graph = self.G.graph
        branch = graph['branch_types'][branch_type]
        new_subgraph = branch['new_subgraph']
        rev = new_subgraph['rev']
        train = new_subgraph['train']

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
        if block_index > 1 and not rev: # need to handle this differently during branch automation
            concat = branch['concat']
            old_concat = next(dsub for dsub in descendants(self.G, subgraph) if graph['concat_type'] in dsub)
            self.split_nconcat(concat, old_concat, subgraph)
            split = branch['split']
            old_split = next(dsub for dsub in descendants(self.G, subgraph) if graph['split_type'] in dsub)
            self.split_nsplit(split, old_split, subgraph)

        # add load links
        try:
            linked_subs = self.linked[base_bridge]
        except KeyError:
            self.linked[base_bridge] = [nbridge]
        else:
            linked_subs.append(nbridge)

        # add graph
        alphas, alpha_edge = graph['alpha_tensor'].split('/')
        alpha_edge = '/'+alpha_edge
        nsub = self.get_new_name(new_subgraph)
        config = new_subgraph['config'].format(block_index).split('_')
        train_set = base_train_set + [nsub, nbridge]
        self.add_subgraph(nsub, config, train, alphas, alpha_edge,
                          block_index, nloss, train_set)

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
            fbridge = next(self.G.successors(forward_subgraph))
            edges.append((fbridge, nloss, {'out': bridge['fout'], 'in': nloss['in']}))
        else:
            b_out = bridge['out']
            if type(b_out) is list:
                b_out = bridge['out'][0] # assumes eval_output is first index

            edges.extend([
                (nbridge, nloss, {'out': b_out, 'in': loss['orig_in']}), 
                (nbridge, old_concat, {'out': b_out, 'in': concat['in'].format(branch_index+2)}),
                (old_split, nloss, {'out': split['out'].format(branch_index+2), 'in': loss['auto_in']}),
            ]) # a_in needs to be attached with split
        self.G.add_edges_from(edges)

        # add graph updates

        graph['loss_tensors'][train].append(nloss+loss['out'])
        if not rev:
            graph['gen_tensor'].append(nbridge+bridge['out'])

        graph['saver_pairs'][nbridge] = nbridge+VARIABLES

        return nloss


    def split_nsplit(self, split, old_split, branch_subgraph):
        successrs = list(self.G.successors(old_split))
        predecessr = next(self.G.predecessors(old_split))
        lss_predecessrs = []
        for sub in successrs:
            pred2 = self.G.predecessors(sub) 
            lss_predecessrs.append(
                next(((sub, s) for s in pred2 if s != old_split), (sub, None))) # dumps branch bridge connecting to the loss function
        
        dloss = next(s for s in lss_predecessrs if s[1] is None)[0]
        dloss_in = self.G.edges[old_split, dloss]['in']
        ancestor_sets = [(rsub, ancestors(self.G, s)) for rsub, s in lss_predecessrs if s is not None]
        new_split_subs = [rsub for rsub, ancestrs in ancestor_sets if branch_subgraph not in ancestrs]

        for sub in new_split_subs:
            new_split = self.get_new_name(split)
            self.G.add_nodes_from([
                (new_split, {'config': ['count{}'.format(self.G.graph['breadth_count']+1)]})])
            new_in = split['in']
            new_out = split['out'].format(1) # always takes the index after real input (index 0)
            old_in = self.G.edges[old_split, sub]['in']
            old_out = self.G.edges[predecessr, old_split]['out']
            self.add_connection(new_split, predecessr, sub,
                                old_in, old_out, new_in, new_out, False, nomod=True)
            self.G.remove_edge(old_split, sub)
            self.G.add_edges_from([
                (new_split, dloss, {'out': split['out'].format(0), 'in': dloss_in})
            ])


    def split_nconcat(self, concat, old_concat, branch_subgraph):
        predecessrs = self.G.predecessors(old_concat)
        successrs = list(self.G.successors(old_concat))
        
        ancestor_sets = [(rsub, ancestors(self.G, rsub)) for rsub in predecessrs]
        new_concat_subs = [rsub for rsub, ancestrs in ancestor_sets if branch_subgraph not in ancestrs]

        for sub in new_concat_subs:
            new_concat = self.get_new_name(concat)
            self.G.add_nodes_from([
                (new_concat, {'config': ['count{}'.format(self.G.graph['breadth_count']+1)]})])
            new_in = concat['in'].format(1) # always takes the index after real input (index 0)
            new_out = concat['out']
            old_in = self.G.edges[old_concat, successrs[0]]['in']
            old_out = self.G.edges[sub, old_concat]['out']
            self.add_connection(new_concat, sub, successrs[0],
                                old_in, old_out, new_in, new_out, False)
            self.G.remove_edge(sub, old_concat)
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
        i = self.G.graph['counts'][name]
        nsub = nsub.format(i)
        self.G.graph['counts'][name] += 1
        return nsub
