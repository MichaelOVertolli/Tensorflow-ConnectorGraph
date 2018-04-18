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

from models.graphs.converter import *
from models.graphs.res_cg_ebm_mad import *
from models.configs.res_cg_ebm_mad import *
from models.model_utils import strip_index
from netgen import *
import networkx as nx
from networkx.algorithms.dag import ancestors, descendants

def test():
    n = NetGen('res_cg_ebm_fr',
               'scaled_began_gmsm_b16_z256_sz128_h128_g0.7_elu_pconv_wxav_alphas',
               'celebc',
               [{'dir': 'base_block', 'data': 'celebc8'},
                {'dir': 'block1', 'data': 'celebc16'},
                {'dir': 'block2', 'data': 'celebc32'},
                {'dir': 'block3', 'data': 'celebc64'},
                {'dir': 'block4', 'data': 'celebc128'}],
               branching=False)
    # n.t_config.max_step = 500
    
    # block_index = 2
    # n.add_subgraphs(block_index)
    
    # subgraph = 'res_gen_pair_0001'
    # base_name = strip_index(subgraph)
    # base_train_set = ancestors(n.G, subgraph)
    # base_bridge = next(n.G.successors(next(n.G.successors(subgraph))))
    # base_train_set = [sub for sub in base_train_set if base_name in sub]
    # base_train_set.append(subgraph)
    # print 'base_bridge: ', base_bridge
    # print 'base_train_set: ', base_train_set
    # print 'counts: ', n.G.graph['counts']
    # print 'loss_tensors: ', n.G.graph['loss_tensors']
    # print 'train_sets: ', n.G.graph['train_sets']
    # print 'gen_tensor: ', n.G.graph['gen_tensor']
    
    # losses = [n.G.graph['loss_tensors']['D'].split('/')[0],
    #           n.G.graph['loss_tensors']['U'][1]]
    # losses.extend([s for s in n.G.successors(base_bridge) if strip_index(losses[0]) in s])
    # nloss = n.add_branch('res_gen_pair', block_index, subgraph, None, base_bridge, base_train_set, 0)
    # losses.append(nloss)
    # print
    # print 'graph extended'
    # print 'counts: ', n.G.graph['counts']
    # print 'loss_tensors: ', n.G.graph['loss_tensors']
    # print 'train_sets: ', n.G.graph['train_sets']
    # print 'gen_tensor: ', n.G.graph['gen_tensor']
    # edges = [' '.join([fr, to, str(n.G.edges[fr, to])]) for fr, to in n.G.edges]
    # sorted(edges)
    # for e in edges:
    #     print e
    # print 'linked: ', n.linked
    # nx.drawing.nx_agraph.write_dot(n.G, './test.dot')
    # print
    # print 'losses: ', losses

    # p = n.get_partial_graph(subgraph, None, losses)
    # nx.drawing.nx_agraph.write_dot(p, './test_partial.dot')

    # print
    # spairs = [pair[1] for pair in p.graph['saver_pairs']]
    # sorted(spairs)
    # for s in spairs:
    #     print s
    # print 'train_sets: ', p.graph['train_sets']['G']
    # print 'loss_tensors: ', p.graph['loss_tensors']
    # print 'img_pairs: ', p.graph['img_pairs']
    # print 'data_inputs: ', p.graph['data_inputs']
    # print 'gen_outputs: ', p.graph['gen_outputs']

    
    # c = n.convert_cg(p, {})
    
    n.run()
    # G = build(config('began_b16_z128_sz32_h128_g0.7_elu_pconv_wxav'))
    

    return n
