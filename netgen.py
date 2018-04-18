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
from models.errors import InvalidBoolMaskError
import json
from netgenrunner import NetGenRunner
import networkx as nx
from networkx.algorithms.dag import ancestors, descendants, topological_sort
from networkx.exception import NetworkXError
from importlib import import_module
from models.graphs.converter import convert
from models.model_utils import strip_index
from models.subgraph import SubGraph
from random import shuffle
import re
from shutil import copytree
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
FETCH = 'fetch_size'
RESIZE = 'resize'
FILTER = 'filter_value'
VARIABLES = '/variables'



class NetGen(object):
    def __init__(self, model_name, model_type, base_dataset, train_program,
                 bool_mask=False, timestamp=None, log_folder=None, branching=True, skip_first=False, skip_n=None,
                 linked={}, base_block_folder=None):

        self.model_name = model_name
        self.model_type = model_type
        
        config = import_module(CONFIG_FILE+model_name)
        graph = import_module(GRAPH_FILE+model_name)
        self.t_ops = import_module(TRAINOPS_FILE+model_name)

        self.config = config.config(model_type)
        self.G = graph.build(self.config)

        if 'frozen' in self.G.graph:
            self.frozen = True
        else:
            self.frozen = False
        
        if timestamp is None:
            self.timestamp = get_time()
        else:
            self.timestamp = timestamp
        if log_folder is None:
            netgen = 'NETGEN'
            if self.frozen:
                netgen += '_frozen'
            log_folder = '_'.join([netgen, model_name, model_type, self.timestamp])
        if bool_mask:
            if not branching:
                raise InvalidBoolMaskError('bool_mask is only valid when NetGen is branching.')
            self.bool_masks = {}
        else:
            self.bool_masks = None
        if branching:
            self.log_dir = os.path.join(LOGS_DIR, base_dataset, 'branching', log_folder)
        else:
            self.log_dir = os.path.join(LOGS_DIR, base_dataset, log_folder)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            for program in train_program:
                os.makedirs(os.path.join(self.log_dir, program[DIR]))
        with open(os.path.join(self.log_dir, 'train_program.txt'), 'w') as f:
            f.write(json.dumps(train_program))
        self.linked = linked
        self.branching = branching
        self.skip_first = skip_first
        if skip_n is not None:
            self.skip_n = skip_n
        else:
            self.skip_n = -1
        self.base_block_folder = base_block_folder
        
        self.train_program = train_program
        self.t_config = trainer_config.config()
        self.t_config.max_step = self.config.alpha_update_steps*2


    def run(self):
        last_index = self.G.graph['last_index']
        base_block = self.G.graph['block_index']
        block_index = base_block
        if self.skip_first:
            base_block = -1
        for program_i, program in enumerate(self.train_program):
            cur_log_dir = os.path.join(self.log_dir, program[DIR])
            if 'glr' in program:
                self.config.g_lr = program['glr']
            if 'dlr' in program:
                self.config.d_lr = program['dlr']
            if 'rlr' in program:
                self.config.r_lr = program['rlr']
            if block_index == base_block:
                load_map = {}
                nx.drawing.nx_agraph.write_dot(
                        self.G, os.path.join(
                            self.log_dir, 'test_full_{}.dot'.format(program[DIR])))
                if self.branching:
                    losses = [sub for sub in self.G.nodes if 'loss' in sub] # don't love this
                    Gpart = self.get_partial_graph(self.G.graph['froot'], self.G.graph['rroot'], losses)
                    nx.drawing.nx_agraph.write_dot(
                        Gpart, os.path.join(
                            self.log_dir, 'test_partial_{}.dot'.format(program[DIR])))
                    part_pickle_path = os.path.join(self.log_dir, 'partial_{}.gpickle'.format(program[DIR]))
                    nx.write_gpickle(Gpart, part_pickle_path)
                    self.run_training(Gpart, cur_log_dir, program, load_map, block_index)
                    if self.bool_masks is not None:
                        self.subset_data(Gpart, program, cur_log_dir, part_pickle_path)
                    del Gpart
                else:
                    self.run_training(self.G, cur_log_dir, program, load_map, block_index)
            else:
                if block_index == last_index:
                    self.t_config.max_step = int(self.t_config.max_step*2.0)
                self.add_subgraphs(block_index)
                if self.branching:
                    branches = []
                    prev_log_dir = os.path.join(self.log_dir, self.train_program[block_index-2][DIR])
                    alpha = self.G.graph['alpha_tensor'].split('/')[0]
                    for subgraph in self.G.nodes:
                        # if you're adding more than one branch sub_index needs to be scaled
                        fr_pair = []
                        try:
                            branch_type = self.G.nodes[subgraph]['branching']
                        except KeyError:
                            continue
                        fr_pair.append((subgraph, branch_type, None))
                        del self.G.nodes[subgraph]['branching'] # node will no longer branch
                        try:
                            rsubgraph, rbranch_type = self.G.nodes[subgraph]['rev_pair']
                        except TypeError:
                            pass
                        else:
                            fr_pair.append((rsubgraph, rbranch_type, subgraph)) # implicit ordering relation guarantees reverse follows forward
                        del self.G.nodes[subgraph]['rev_pair']
                        branches.append(fr_pair)
                    shuffle(branches) # need to prevent ordering effects
                    branches = [item for fr_pair in branches for item in fr_pair] # guarantees that forward branch completed before reverse post shuffle
                    for subgraph, branch_type, fsubgraph in branches:
                        # implement autobranch here
                        # bridge_name = strip_index(self.G.graph['branch_types'][branch_type]['bridge']['name'])
                        rev = self.G.graph['branch_types'][branch_type]['new_subgraph']['rev']
                        if rev:
                            ds = topological_sort(nx.subgraph(self.G, descendants(self.G, subgraph)))
                            base_train_set = []
                            for d in ds:
                                if branch_type in d:
                                    base_train_set.append(d)
                                else:
                                    break
                            new_rsub = next(s for s in self.G.predecessors(subgraph) if alpha not in s) # need to ignore alphas
                            # new_fsub = next(self.G.successors(fsubgraph)) # forward subgraph added by add_subgraphs
                            # self.G.add_nodes_from([
                            #     (new_fsub, {
                            #         'rev_pair': [new_rsub, branch_type], # adds reverse details to matching forward_subgraph
                            #     })])
                            base_bridge = next(s for s in self.G.predecessors(new_rsub) if alpha not in s) # for loading in linked_subs
                            losses = [s for s in self.G.successors(self.G.nodes[base_bridge]['paired'])
                                      if strip_index('loss') in s] # will output regular and reverse losses
                        else:
                            base_train_set = list(ancestors(self.G, subgraph))
                            new_sub = next(self.G.successors(subgraph)) # subgraph added by add_subgraphs
                            fbridge = next(self.G.successors(new_sub))
                            if self.G.graph['reversible']:
                                new_rsub = next(self.G.successors(self.G.nodes[fbridge]['paired']))
                                rbranch_type = self.G.graph['branch_types'][branch_type]['rev_type']
                                self.G.add_nodes_from([
                                    (new_sub, {
                                        'branching': branch_type,
                                        'rev_pair': [new_rsub, rbranch_type],
                                    })])
                            else:
                                self.G.add_nodes_from([
                                    (new_sub, {
                                        'branching': branch_type,
                                        'rev_pair': None,
                                    })])
                            print
                            print new_sub, self.G.nodes[new_sub]
                            print
                            base_bridge = next(self.G.successors(new_sub))
                            losses = [s for s in self.G.successors(base_bridge) if 'loss' in s] # will output regular and reverse losses
                        # base_bridge = next(sub for sub in base_train_set if bridge_name in sub)
                        base_train_set.append(subgraph)
                        base_train_set = [sub for sub in base_train_set if branch_type in sub]
                        losses.extend([self.G.graph['loss_tensors']['D'].split('/')[0],
                                       self.G.graph['loss_tensors']['U'][1]])
                        for i in range(self.G.graph['breadth_count'] - 1):
                            nlosses = self.add_branch(branch_type, block_index, subgraph, fsubgraph,
                                                      base_bridge, base_train_set, i)
                            losses.extend(nlosses)
                        nx.drawing.nx_agraph.write_dot(
                            self.G, os.path.join(
                                self.log_dir, 'test_full_{}_{}.dot'.format(program[DIR], subgraph)))
                        if self.G.graph['reversible']:
                            if fsubgraph is not None:
                                Gpart = self.get_partial_graph(fsubgraph, subgraph, losses)
                                prev_fsubgraph = next(s for s in Gpart.predecessors(fsubgraph) if alpha not in s) 
                                branched_prev_log_dir = os.path.join(prev_log_dir, prev_fsubgraph)
                                branched_cur_log_dir = os.path.join(cur_log_dir, fsubgraph)
                            else:
                                continue
                        else:
                            Gpart = self.get_partial_graph(subgraph, None, losses)
                            prev_subgraph = next(s for s in Gpart.predecessors(subgraph) if alpha not in s) 
                            branched_prev_log_dir = os.path.join(prev_log_dir, prev_subgraph)
                            branched_cur_log_dir = os.path.join(cur_log_dir, subgraph)
                        nx.drawing.nx_agraph.write_dot(
                            Gpart, os.path.join(
                                self.log_dir, 'test_partial_{}_{}.dot'.format(program[DIR], subgraph)))
                        edges = [' '.join([fr, to, str(Gpart.edges[fr, to])]) for fr, to in Gpart.edges]
                        sorted(edges)
                        for e in edges:
                            print e
                        if block_index > (base_block + 1):
                            load_map = self.build_load_map(branched_prev_log_dir, None)
                        else:
                            load_map = self.build_load_map(prev_log_dir, None)
                        # for key in load_map:
                        #     print key, load_map[key]
                        part_pickle_path = os.path.join(self.log_dir, 'partial_{}_{}.gpickle'.format(program[DIR], subgraph))
                        nx.write_gpickle(Gpart, part_pickle_path)

                        try:
                            bool_mask = self.bool_masks[fsubgraph]
                        except KeyError:
                            bool_mask = self.bool_masks[subgraph]
                        except TypeError:
                            bool_mask = None
                        self.run_training(Gpart, branched_cur_log_dir, program, load_map, block_index, bool_mask)
                        if bool_mask is not None:
                            self.subset_data(Gpart, program, branched_cur_log_dir, part_pickle_path)
                            self.save_subset(branched_cur_log_dir, bool_mask)
                        del Gpart
                else:
                    if self.base_block_folder is not None:
                        prev_log_dir = os.path.join(self.base_block_folder, self.train_program[block_index-1][DIR])
                    else:
                        prev_log_dir = os.path.join(self.log_dir, self.train_program[block_index-1][DIR])
                    load_map = self.build_load_map(prev_log_dir)
                    nx.drawing.nx_agraph.write_dot(
                        self.G, os.path.join(
                            self.log_dir, 'test_full_{}.dot'.format(program[DIR])))
                    self.run_training(self.G, cur_log_dir, program, load_map, block_index)
            block_index += 1
            # after a training step finishes there should be no linked subgraphs
            self.linked = {}
            try:
                frozen = self.G.graph['frozen']
            except KeyError:
                pass
            else:
                for type_ in frozen:
                    subs = [s for s in self.G.nodes if type_ in s]
                    for sub in subs:
                        try:
                            _ = self.G.nodes[sub]['frozen']
                        except KeyError:
                            self.freeze_subgraph(sub, cur_log_dir)

        nx.write_gpickle(self.G, os.path.join(self.log_dir, 'final_graph.gpickle'))


    def save_subset(self, branch_dir, bool_mask):
        bools = np.concatenate(bool_mask)
        indexes = []
        for i in range(bools.shape[0]):
            if bools[i]:
                indexes.append(i)
        with open(os.path.join(branch_dir, 'data_subset_indexes.json'), 'w') as f:
            f.write(json.dumps(indexes))
        name = branch_dir.split('/')[-1]
        count = str(np.sum(bool_mask))
        with open(os.path.join(self.log_dir, 'subset_counts.txt'), 'a') as f:
            f.write(' '.join([name, count, '\n']))


    def get_partial_graph(self, fsubgraph, rsubgraph, losses):
        # creating partial graph
        base = [fsubgraph, self.G.graph['alpha_tensor'].split('/')[0]]
        base.extend(losses)
        # g_losses = self.G.graph['train_sets']['G'].keys()
        bot_loss = next(l for l in losses if l in self.G.graph['train_sets']['G'])
        # for loss in bot_losses:
        #     subs = [s for s in self.G.predecessors(loss) if self.G.graph['split_type'] in s]
        #     if len(subs) > 0:
        #         bot_loss = loss
        #         break
        try:
            top = set(descendants(self.G, rsubgraph)) | set(ancestors(self.G, rsubgraph))
        except NetworkXError:
            top = set(ancestors(self.G, fsubgraph))
        else:
            base.append(rsubgraph)
            top &= set(ancestors(self.G, fsubgraph))
        descendnts = set(descendants(self.G, fsubgraph))
        bot = descendnts & set(ancestors(self.G, bot_loss))
        full = list(top | bot)+base
        partial_graph = nx.subgraph(self.G, full).copy()
        partial_graph.graph = deepcopy(self.G.graph)

        graph = partial_graph.graph

        reversible = graph['reversible']

        # adjusting graph attributes
        saver_pairs = {}
        for sub in partial_graph.nodes:
            try:
                variables = graph['saver_pairs'][sub]
            except KeyError:
                continue
            else:
                saver_pairs[sub] = variables
        graph['saver_pairs'] = saver_pairs.items()
        gloss_tensors = []
        uloss_tensors = {}
        concat = next(s for s in descendnts if graph['concat_type'] in s)
        bridge_base, uloss, outn = graph['loss_tensors']['U']
        outp = graph['loss_tensors']['G']
    
        for loss in [k for k in graph['train_sets']['G']]:
            if loss not in losses:
                del graph['train_sets']['G'][loss]
            else:
                gloss_tensors.append(loss+outp)
                bridge = next(s for s in partial_graph.predecessors(loss) if bridge_base in s)
                index = int(re.search('\d+(?=\:)', partial_graph.edges[bridge, concat]['in']).group())
                uloss_tensors[loss] = uloss+outn.format(index-1)
        graph['loss_tensors']['G'] = gloss_tensors
        graph['loss_tensors']['U'] = uloss_tensors

        if reversible:
            rloss_tensors = []
            routp = graph['loss_tensors']['R']
            for loss in [k for k in graph['train_sets']['R']]:
                if loss not in losses:
                    del graph['train_sets']['R'][loss]
                else:
                    rloss_tensors.append(loss+routp)
            graph['loss_tensors']['R'] = rloss_tensors

        
        split = next(s for s in descendants(partial_graph, concat) if graph['split_type'] in s) # there should only ever be one split post concat
        graph['a_output'] = split+graph['a_output']
        
        img_pairs = []
        gen_outputs = []
        if reversible:
            rev_inputs = {}
            rgen_outputs = []
        for i, sub in enumerate(partial_graph.successors(fsubgraph)):
            bridge = next(partial_graph.successors(sub))
            bridge_out = bridge+graph['img_pairs']['G'][1]
            img_pairs.append(('G{}'.format(i), bridge_out))
            gen_outputs.append(bridge_out)
            if reversible:
                rbridge_out = bridge+graph['img_pairs']['R'][1]
                img_pairs.append(('R{}'.format(i), rbridge_out))
                rgen_outputs.append(rbridge_out)
                # loss = next(s for s in partial_graph.successors(bridge)
                #             if 'loss' in s and len(partial_graph.nodes[s]['in']) > 1) # only reverse loss has 2 inputs from one bridge
                # rbridge_in = next(s for s in graph['train_sets'][loss]
                #                   if graph['img_pairs']['R'][0] in s) # reverse loss allows us to find corresponding rev bridge
                rbridge_in = partial_graph.nodes[bridge]['paired']
                rev_inputs[bridge_out] = rbridge_in+graph['rev_inputs']
        for sub in partial_graph.predecessors(losses[-1]): # not Disc or mad loss
            if graph['img_pairs']['A_'][0] in sub:
                for i in range(1, graph['breadth_count']+1):
                    img_pairs.append(('A_G{}'.format(i), sub+graph['img_pairs']['A_'][1].format(i)))
                img_pairs.append(('A_D', sub+graph['img_pairs']['A_'][1].format(0)))
                break # there's only one relevant subgraph
        graph['img_pairs'] = img_pairs
        graph['gen_outputs']['G'] = gen_outputs
        graph['gen_tensor'] = gen_outputs
        if reversible:
            graph['rev_inputs'] = rev_inputs
            graph['gen_outputs']['R'] = rgen_outputs

        data_inputs = []
        data_inputs.append(graph['data_inputs']['loss'])
        for sub in descendants(partial_graph, fsubgraph):
            if graph['concat_type'] in sub:
                data_inputs.append(sub+graph['data_inputs']['concat'])
                break
        graph['data_inputs'] = data_inputs

        # add concat/split for rev
        if reversible:
            concat = graph['branch_types'][strip_index(rsubgraph)]['concat']
            new_concat = self.get_new_name(concat)
            split = graph['branch_types'][strip_index(rsubgraph)]['split']
            new_split = self.get_new_name(split)
            alpha = graph['alpha_tensor'].split('/')[0]
            predecessrs = [s for s in partial_graph.predecessors(rsubgraph) if alpha not in s]
            
            partial_graph.add_nodes_from([
                (new_concat, {'config': ['count2']})])
            partial_graph.add_nodes_from([
                (new_split, {'config': ['count2']})])

            c_in = partial_graph.edges[predecessrs[0], rsubgraph]['in']
            
            outp, out2 = graph['branch_types'][strip_index(fsubgraph)]['new_subgraph']['out']
            inpt, inp2 = graph['branch_types'][strip_index(fsubgraph)]['new_subgraph']['in']

            if self.frozen and fsubgraph != self.G.graph['froot']:
                outp = '/'.join(['', fsubgraph+outp])
                out2 = '/'.join(['', fsubgraph+out2])
            edges = []

            edges.extend([
                (new_concat, rsubgraph, {'out': concat['out'], 'in': c_in}),
                (fsubgraph, new_split, {'out': out2, 'in': split['in']}),
            ])

            for i, subgraph in enumerate(predecessrs):
                bridge = next(s for s in partial_graph.predecessors(subgraph) if alpha not in s)
                paired_subgraph = next(partial_graph.predecessors(partial_graph.nodes[bridge]['paired']))

                # if self.branching and fsubgraph != self.G.graph['froot']:
                #     inpt = '/'.join([paired_subgraph, inpt])
                #     inp2 = '/'.join([paired_subgraph, inp2])
                
                old_out = partial_graph.edges[subgraph, rsubgraph]['out']
                new_in = concat['in'].format(i)
                new_out = split['out'].format(i)

                edges.extend([
                    (subgraph, new_concat, {'out': old_out, 'in': new_in}),
                    (new_split, paired_subgraph, {'out': new_out, 'in': inp2}),
                ])

                partial_graph.edges[fsubgraph, paired_subgraph]['in'] = inpt
                partial_graph.edges[fsubgraph, paired_subgraph]['out'] = outp

                partial_graph.remove_edge(subgraph, rsubgraph)

            partial_graph.add_edges_from(edges)
        
        return partial_graph


    def subset_data(self, G, program, model_dir, gpickle):
        print 'Generating data subsets.'
        runner = NetGenRunner(self.model_name, self.model_type,
                              model_dir,
                              [(gpickle, G)], branching=self.branching)
        runner.prep_sess(G, program[DATA], program[FETCH], program[RESIZE], greyscale=G.graph['greyscale'])
        losses, subgraphs = runner.run_losses(G, self.config.batch_size)
        bool_sets = runner.subset_data(losses, program[FILTER])
        runner.close()
        del runner
        print
        print 'Bool masks: ', subgraphs
        print
        for sub, bool_mask in zip(subgraphs, bool_sets):
            self.bool_masks[sub] = bool_mask


    def build_load_map(self, prev_log_dir, cur_log_dir=None):
        load_map = {}
        subgraphs = [(prev_log_dir, s)
                     for s in os.walk(prev_log_dir).next()[1]]
        try:
            current = os.walk(cur_log_dir).next()[1]
        except TypeError:
            pass
        else:
            subgraphs = [pair for pair in subgraphs if pair[1] not in current]
            subgraphs += [(cur_log_dir, s) for s in current]
        for log_dir, subgraph in subgraphs:
            try:
                frozen = self.G.nodes[subgraph]['frozen']
            except KeyError:
                frozen = None
            load_map[subgraph] = [
                os.path.join(log_dir, subgraph),
                None,
                None,
                frozen,
            ]
            try:
                linked_subs = self.linked[subgraph]
            except KeyError:
                pass
            else:
                for lsub in linked_subs:
                    load_map[lsub] = [
                        os.path.join(log_dir, lsub),
                        subgraph,
                        os.path.join(log_dir, subgraph),
                        frozen,
                    ]
        return load_map


    def convert_cg(self, G, load_map, log_folder): 
        conngraph, inputs, outputs = convert(G, self.config, load_map)

        conngraph = self.t_ops.build_train_ops(log_folder, conngraph, inputs, outputs, **G.graph)

        get_feed_dict = self.t_ops.build_feed_func(**G.graph)
        conngraph.attach_func(get_feed_dict)
        send_outputs = self.t_ops.build_send_func(**G.graph)
        conngraph.attach_func(send_outputs)

        return conngraph


    def run_training(self, G, log_folder, program, load_map, block_index, bool_mask=None):
        if block_index < self.skip_n:
            return
        else:
            self.base_block_folder = None # allows subsequent growth to generate from normal place
        full_log_folder = os.path.join(log_folder, get_time())
        conngraph = self.convert_cg(G, load_map, full_log_folder)
        conngraph.set_block_index(block_index)
        trainer = Trainer(self.model_name, self.model_type, self.t_config,
                          program[DATA], program[FETCH], program[RESIZE],
                          bool_mask, full_log_folder, conngraph, True, G.graph['greyscale'])
        # if block_index > 0:
        #     weights = trainer.sess.graph.get_tensor_by_name('res_gen_pair_08/G/Conv_1/weights:0')
        #     w = trainer.sess.run(weights)
        #     print w[0, 0, :, :]
        #     return w
        step = trainer.train()
        # if block_index == 0:
        #     tester = trainer.sess.graph.get_operation_by_name('tester1234')
        #     weights = trainer.sess.graph.get_tensor_by_name('res_gen_pair_08/G/Conv_1/weights:0')
        #     tester.run(session=trainer.sess)
        #     w = trainer.sess.run(weights)
        #     print w[0, 0, :, :]
        conngraph.save_subgraphs(log_folder, step, trainer.sess)
        trainer.close()
        del trainer
        del conngraph


    def split_dataset(self, trainer, dataset):
        feed_dict = trainer.c_graph.get_feed_dict(trainer)
        v = trainer.sess.run(G.graph['loss_tensors']['R'], feed_dict)


    def freeze_subgraph(self, subgraph, log_folder):
        passed_train_block = False
        for program in self.train_program: # copy saved subgraph folders
            if passed_train_block:
                copytree(os.path.join(log_folder, subgraph), os.path.join(self.log_dir, program[DIR], subgraph))
            elif subgraph in os.listdir(os.path.join(self.log_dir, program[DIR])):
                passed_train_block = True
            else:
                continue
        graph = self.G.graph
        for key in graph['train_sets']: # remove frozen graphs from training graph sets
            if self.branching and key != 'D':
                for loss in graph['train_sets'][key]:
                    graph['train_sets'][key][loss] = [s for s in graph['train_sets'][key][loss] if s != subgraph]
            else:
                graph['train_sets'][key] = [s for s in graph['train_sets'][key] if s != subgraph]
        if self.branching: # remove frozen graphs from saving graph set
            del graph['saver_pairs'][subgraph]
        else:
            graph['saver_pairs'] = [p for p in graph['saver_pairs'] if p[0] != subgraph]
        if subgraph in graph['gen_input']:
            graph['gen_input'] = '/'.join([subgraph, graph['gen_input']])
        attributes = {'frozen': True}
        try:
            outputs = self.G.nodes[subgraph]['outputs']
        except KeyError:
            pass
        else:
            try:
                outputs = '/'.join(['', subgraph+outputs])
            except TypeError:
                outputs = ['/'.join(['', subgraph+o]) for o in outputs]
            attributes['outputs'] = outputs
        try:
            inputs = self.G.nodes[subgraph]['inputs']
        except KeyError:
            pass
        else:
            try:
                inputs = '/'.join(['', subgraph+inputs])
            except TypeError:
                inputs = ['/'.join(['', subgraph+i]) for i in inputs]
            attributes['inputs'] = inputs
        self.G.add_nodes_from([
            (subgraph, attributes),
        ])
        for psub in self.G.predecessors(subgraph):
            in_ = self.G.edges[psub, subgraph]['in']
            outpts = self.G.edges[psub, subgraph]['out']
            try:
                inpts = '/'.join(['', subgraph+in_])
            except TypeError:
                inpts = ['/'.join(['', subgraph+i]) for i in in_]
            self.G.add_edges_from([
                (psub, subgraph, {'out': outpts, 'in': inpts})
            ])
        for ssub in self.G.successors(subgraph):
            inpts = self.G.edges[subgraph, ssub]['in']
            out_ = self.G.edges[subgraph, ssub]['out']
            try:
                outpts = '/'.join(['', subgraph+out_])
            except TypeError:
                outpts = ['/'.join(['', subgraph+o]) for o in out_]
            self.G.add_edges_from([
                (subgraph, ssub, {'out': outpts, 'in': inpts})
            ])


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
        # 
        if self.branching:
            self.G.graph['saver_pairs'][new_subgraph] = new_subgraph+VARIABLES
        else:
            self.G.graph['saver_pairs'].append((new_subgraph, new_subgraph+VARIABLES))
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
        try:
            branch_types = self.G.graph['branch_types']
        except KeyError:
            pass
        alphas, alpha_edge = self.G.graph['alpha_tensor'].split('/')
        alpha_edge = '/'+alpha_edge
        subgraphs = []
        for prev_subgraph, next_subgraph in [e for e in self.G.edges]:
            try:
                key = self.G.edges[prev_subgraph, next_subgraph]['mod']
            except KeyError:
                continue
            try:
                new_subgraph = growth_types[key]
            except KeyError:
                new_subgraph = branch_types[key]['new_subgraph']
                branch = True
            else:
                branch = False

            nsub = self.get_new_name(new_subgraph)
            
            config = new_subgraph['config'].format(block_index).split('_')
            train = new_subgraph['train']

            rev = new_subgraph['rev']
            if rev: # applies to generator and discriminator rev graphs
                if branch:
                    paired = self.G.nodes[prev_subgraph]['paired'] # prev should be the bridge if it's reversed
                    loss = next(s for s in self.G.successors(paired)
                                if 'loss' in s and type(self.G.edges[paired, s]['out']) is list) # only reverse loss has 2 inputs from one bridge
                    # for sub in self.G.nodes:
                    #     try:
                    #         rev_pair = self.G.nodes[sub]['rev_pair'][0]
                    #     except (KeyError, TypeError):
                    #         continue
                    #     if rev_pair == next_subgraph:
                    #         loss = next(
                else:
                    loss = None
            else:
                loss = next((s for s in self.G.successors(next_subgraph)
                             if 'loss' in s and type(self.G.edges[next_subgraph, s]['out']) is not list), None)

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
        try:
            config = bridge['config']
        except KeyError:
            bridge_attrs = {}
        else:
            bridge_attrs = {'config': config}
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
        try:
            config = loss['config']
        except KeyError:
            pass
        else:
            loss_attrs['config'] = config

        self.G.add_nodes_from([
            (nbridge, bridge_attrs),
            (nloss, loss_attrs),
        ])

        # add concat
        if not rev: 
            concat = branch['concat']
            old_concat = next(dsub for dsub in descendants(self.G, subgraph) if graph['concat_type'] in dsub)
            split = branch['split']

            # old_split = next(dsub for dsub in descendants(self.G, subgraph) if graph['split_type'] in dsub)
            cur_bridge = next(self.G.successors(next(self.G.successors(subgraph))))
            self.G.edges[cur_bridge, old_concat]['in'] = concat['in'].format(1) # re-map existing bridge to 2nd index of concat
            cur_loss = next(sub for sub in self.G.successors(cur_bridge)
                            if sub != old_concat and type(self.G.edges[cur_bridge, sub]['out']) is not list)
            old_split = next(sub for sub in self.G.predecessors(cur_loss) if cur_bridge != sub)
            
            if block_index > 1: # need to handle this differently during branch automation
                self.split_nconcat(concat, old_concat, subgraph)
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
        if rev:            
            paired = None
            for new_fsub in self.G.successors(forward_subgraph):
                bridge_ = next(self.G.successors(new_fsub))
                if self.G.nodes[bridge_]['paired'] is None: # if > 2 branches, this will grab the first un-paired bridge
                    paired = bridge_
                    break
            if paired is None:
                raise TypeError('No valid paired bridge was found.')
            self.G.add_nodes_from([
                (new_fsub, {
                    'rev_pair': [nsub, branch_type], # adds reverse details to matching forward_subgraph
                })])
            self.G.add_nodes_from([
                (nbridge, {
                    'paired': paired,}),
                (paired, {
                    'paired': nbridge,}),
            ])
            print
            print new_fsub, self.G.nodes[new_fsub]
            print
        else:
            self.G.add_nodes_from([
                (nsub, {
                    'branching': branch_type,
                    'rev_pair': None, # will be assigned when the reverse subgraph is added
                })])
            self.G.add_nodes_from([
                (nbridge, {
                    'paired': None,})]) # will be assigned when the reverse subgraph is added


        print
        print nsub, self.G.nodes[nsub]
        print 
        # add connections
        if rev:
            prev_subgraph = nbridge
            next_subgraph = subgraph
            old_in = bridge['in']
            old_out = new_subgraph['out'] # assumes that all branches are identical to their branch points which is reasonable
            if self.frozen:
                try:
                    old_in = '/'.join(['', subgraph+old_in])
                except TypeError:
                    old_in = ['/'.join(['', subgraph+i]) for i in old_in]
        else:
            prev_subgraph = subgraph
            next_subgraph = nbridge
            old_in = bridge['in']
            old_out = new_subgraph['out'] # assumes that all branches are identical to their branch points which is reasonable
            if self.frozen:
                try:
                    old_out = '/'.join(['', subgraph+old_out])
                except TypeError:
                    old_out = ['/'.join(['', subgraph+o]) for o in old_out]
        new_in = new_subgraph['in']
        new_out = new_subgraph['out']
        self.add_connection(nsub, prev_subgraph, next_subgraph,
                            old_in, old_out, new_in, new_out, rev)

        
        edges = []
        if rev:
            fbridge = self.G.nodes[nbridge]['paired']
            edges.append((fbridge, nloss, {'out': bridge['fout'], 'in': loss['in']}))
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

        try:
            out = [s for s in self.G.successors(fbridge) if 'loss' in s] # will output regular and reverse losses
        except UnboundLocalError:
            out = [nloss] # will output regular and reverse losses
        # add graph updates

        # graph['loss_tensors'][train].append(nloss+loss['out'])
        # if not rev:
        #     graph['gen_tensor'].append(nbridge+bridge['out'])

        graph['saver_pairs'][nbridge] = nbridge+VARIABLES

        return out


    def split_nsplit(self, split, old_split, branch_subgraph):
        successrs = list(self.G.successors(old_split))
        predecessr = next(self.G.predecessors(old_split))
        lss_predecessrs = []
        for sub in successrs:
            pred2 = self.G.predecessors(sub) 
            lss_predecessrs.append(
                next(((sub, s) for s in pred2 if self.G.graph['split_type'] not in s), (sub, None))) # dumps branch bridge connecting to the loss function

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
                                old_in, old_out, new_in, new_out, False, nomod=True)
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


    def close(self):
        del self.model_name
        del self.model_type
        
        del self.t_ops

        del self.config
        self.G.clear()
        del self.G

        del self.frozen
        del self.timestamp
        del self.log_dir

        try:
            for k in self.bool_masks:
                del self.bool_masks[k][:]
        except TypeError:
            pass
        else:
            self.bool_masks.clear()
        del self.bool_masks

        for k in self.linked:
            del self.linked[k][:]
        self.linked.clear()
        del self.linked
        
        del self.branching
        
        for program in self.train_program:
            program.clear()
        del self.train_program[:]
        del self.train_program
        
        del self.t_config
