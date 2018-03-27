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

# convert to using new net subgraphs (pair, rev)

#Models
GNF0 = 'res_gen_pair_0000'
GNF1 = 'res_gen_pair_0001'
GNF2 = 'res_gen_pair_0002'
BRF0 = 'res_bridge_00'
BRF1 = 'res_bridge_01'
DSC0 = 'res_rev_0000'
DSC1 = 'res_rev_0001'
BRGD = 'res_bridge_02'
DGN0 = 'res_gen_pair_0003'
DGN1 = 'res_gen_pair_0004'
BRGU = 'res_bridge_03'
REST = 'res_train'
NCN0 = 'mad_nconcat_00'
NSP0 = 'mad_nsplit_00'
LSG0 = 'cqs_loss_set_00'
LSG1 = 'cqs_loss_set_01'
LSSD = 'cqs_loss_set_02'
LSSU = 'mad_loss_0'
ALIN = 'res_alphas_0'

#Inputs
INPT = '/input:0'
INP2 = '/input2:0'
INPN = '/input{}:0'
G_IN = '/gen_input:0'
D_IN = '/data_input:0'
M_IN = '/mix_input:0'
O_IN = '/orig_input:0'
A_IN = '/autoencoded_input:0'

#Outputs
OUTP = '/output:0'
OUT2 = '/output2:0'
OUTN = '/output{}:0'
GOUT = '/gen_output:0'
DOUT = '/data_output:0'
MOUT = '/mix_output:0'

#Alphas
ALPH = '/alpha{}:0'

#Variables
VARS = '/trainable_variables'
VARIABLES = '/variables'

BCOUNT = 2

def build(config):
    G = nx.DiGraph(
        train_scope='res_train',
        reversible=False,
        greyscale=False,
        concat_type='mad_nconcat',
        split_type='mad_nsplit',
        froot=GNF0,
        rroot=None,
        block_index=1,
        last_index=4,
        counts={
            'res_bridge': 4,
            'res_rev': 2,
            'res_gen_pair': 5,
            'cqs_loss_set': 3,
            'mad_nconcat': 1,
            'mad_nsplit': 1,},
        breadth_count=BCOUNT,
        loss_tensors={
            'G': OUTP, # needs to be modified with partial graph (done)
            'D': LSSD+OUTP,
            'U': ['res_bridge', LSSU, OUTN],}, # [LSSU+OUTN.format(i) for i in range(BCOUNT)],},
        train_sets={
            'G': {
                LSG0: [GNF0, GNF1, BRF0], # needs to be modified with partial graph (done)
                LSG1: [GNF0, GNF2, BRF1]}, # needs to be modified with partial graph (done)
            'D': [DSC0, DSC1, DGN0, DGN1, BRGD, BRGU],},
        img_pairs={
            'G': ['res_bridge', OUTP],
            'R': ['res_bridge', OUT2],
            'A_': ['mad_nsplit', OUTN],
            },
            # ('G1', BRF0+OUTP), # need to be modified when we generate a partial graph (done)
            # ('G2', BRF1+OUTP),
            # ('A_G1', NSP0+OUTN.format(1)),
            # ('A_G2', NSP0+OUTN.format(2)),
            # ('A_D', NSP0+OUTN.format(0)),],
        saver_pairs=dict([
            (GNF0, GNF0+VARIABLES), # should be modified when making partial graph (done)
            (GNF1, GNF1+VARIABLES),
            (GNF2, GNF2+VARIABLES),
            (BRF0, BRF0+VARIABLES),
            (BRF1, BRF1+VARIABLES),
            (BRGD, BRGD+VARIABLES),
            (DSC0, DSC0+VARIABLES),
            (DSC1, DSC1+VARIABLES),
            (DGN0, DGN0+VARIABLES),
            (DGN1, DGN1+VARIABLES),
            (BRGU, BRGU+VARIABLES),]),
        gen_tensor=[],# [BRF0+OUTP, BRF1+OUTP], # should be modified when making partial graphs (done)
        gen_input=GNF0+INPT, # should be generic across all partial graphs
        rev_inputs=None, # should be modified when making partial graphs (done)
        data_inputs={
            'loss': LSSD+O_IN,
            'concat': INPN.format(0),
            },
            # [
            # NCN0+INPN.format(0), add this when making partial graph (done)
            # LSSD+O_IN,],
        alpha_tensor=ALIN+ALPH,
        gen_outputs={
            'G': []},# [BRF0+OUTP, BRF1+OUTP]}, # should be modified when making partial graph (done)
        a_output=OUTN.format(0), # should be modified when making partial graphs (done)
        branch_types={
            'res_gen_pair': {
                'new_subgraph': {
                    'name': 'res_gen_pair_{:04}', # add leading zeros up to 4
                    'in': INPT, #INP2],
                    'out': OUTP, #OUT2],
                    'config': 'block{}_clone',
                    'train': 'G',
                    'rev': False,
                },
                'bridge': {
                    'name': 'res_bridge_{:02}',
                    'in': INPT, #INP2],
                    'out': OUTP, #OUT2],    eval_output is always the first index
                    'config': ['to_image', 'clone'],
                    'outputs': [OUTP], #OUT2],
                },
                'loss': {
                    'name': 'cqs_loss_set_{:02}',
                    'orig_in': O_IN,
                    'auto_in': A_IN,
                    'out': OUTP,
                    'outputs': [OUTP],
                },
                'concat': {
                    'name': 'mad_nconcat_{:02}',
                    'in': INPN,
                    'out': OUTP,
                },
                'split': {
                    'name': 'mad_nsplit_{:02}',
                    'in': INPT,
                    'out': OUTN,
                },
            },
        },
        growth_types={
            BRGD: {
                'name': 'res_rev_{:04}',
                'in': INPT,
                'out': OUTP,
                'config': 'block{}',
                'train': 'D',
                'rev': True,
            },
            BRGU: {
                'name': 'res_gen_pair_{:04}',
                'in': INPT,
                'out': OUTP,
                'config': 'block{}',
                'train': 'D',
                'rev': False,
            },
        },
    )
    
    alpha_inputs = []
    for i in range(config.repeat_num-1):
        alpha_inputs.append(ALPH.format(i))
    
    # Input SubGraphs    
    G.add_nodes_from([
        (GNF0, {'inputs': [INPT]}),
        (NCN0, {'inputs': [INPN.format(0)]}),
        (LSSD, {'inputs': [O_IN]}),
        (ALIN, {'inputs': alpha_inputs}),
    ])
    # Output SubGraphs
    G.add_nodes_from([
        (BRF0, {'outputs': [OUTP]}),
        (BRF1, {'outputs': [OUTP]}),
        (LSG0, {'outputs': [OUTP]}),
        (LSG1, {'outputs': [OUTP]}),
        (LSSD, {'outputs': [OUTP]}),
        (LSSU, {'outputs': [OUTN.format(i) for i in range(G.graph['breadth_count'])]}),
    ])
    # SubGraphs with config mods
    G.add_nodes_from([
        (BRF0, {'config': ['to_image', 'clone']}),
        (BRF1, {'config': ['to_image', 'clone']}),
        (BRGU, {'config': ['to_image']}),
        (GNF0, {'config': ['block0', 'clone', 'base']}),
        (GNF1, {'config': ['block1', 'clone']}),
        (GNF2, {'config': ['block1', 'clone']}),
        (DSC0, {'config': ['block0', 'base']}),
        (DSC1, {'config': ['block1']}),
        (DGN0, {'config': ['block0', 'base']}),
        (DGN1, {'config': ['block1']}),
        (NCN0, {'config': ['count{}'.format(G.graph['breadth_count']+1)]}), # breadth_count + real_input #### NEED one per branch_set
        (NSP0, {'config': ['count{}'.format(G.graph['breadth_count']+1)]}),
        (LSSU, {'config': ['count{}'.format(G.graph['breadth_count']+1)]}),
    ])
    # Branching SubGraphs
    G.add_nodes_from([
        (GNF1, {
            'branching': 'res_gen_pair',
            'rev_pair': None}),
        (GNF2, {
            'branching': 'res_gen_pair',
            'rev_pair': None}),
    ])
    # Remaining inner SubGraphs
    # None

    G.add_edges_from([
        # Main path
        (GNF0, GNF1, {'out': OUTP, 'in': INPT}),
        (GNF0, GNF2, {'out': OUTP, 'in': INPT}),
        (GNF1, BRF0, {'out': OUTP, 'in': INPT, 'mod': 'res_gen_pair'}),
        (GNF2, BRF1, {'out': OUTP, 'in': INPT, 'mod': 'res_gen_pair'}),
        (BRF0, NCN0, {'out': OUTP, 'in': INPN.format(1)}), # 0 is for the real_input
        (BRF1, NCN0, {'out': OUTP, 'in': INPN.format(2)}),
        (NCN0, LSSU, {'out': OUTP, 'in': INPT}),
        (NCN0, BRGD, {'out': OUTP, 'in': INPT}),
        (BRGD, DSC1, {'out': OUTP, 'in': INPT, 'mod': BRGD}),
        (DSC1, DSC0, {'out': OUTP, 'in': INPT}),
        (DSC0, DGN0, {'out': OUTP, 'in': INPT}),
        (DGN0, DGN1, {'out': OUTP, 'in': INPT}),
        (DGN1, BRGU, {'out': OUTP, 'in': INPT, 'mod': BRGU}),
        (BRGU, NSP0, {'out': OUTP, 'in': INPT}),
        (BRF0, LSG0, {'out': OUTP, 'in': O_IN}),
        (BRF1, LSG1, {'out': OUTP, 'in': O_IN}),
        (NSP0, LSSD, {'out': OUTN.format(0), 'in': A_IN}),
        (NSP0, LSG0, {'out': OUTN.format(1), 'in': A_IN}),
        (NSP0, LSG1, {'out': OUTN.format(2), 'in': A_IN}),
        # Alphas
        (ALIN, GNF0, {'out': ALPH.format(0), 'in': ALPH.format(0)}),
        (ALIN, GNF1, {'out': ALPH.format(1), 'in': ALPH.format(1)}),
        (ALIN, GNF2, {'out': ALPH.format(1), 'in': ALPH.format(1)}),
        (ALIN, DSC0, {'out': ALPH.format(0), 'in': ALPH.format(0)}),
        (ALIN, DSC1, {'out': ALPH.format(1), 'in': ALPH.format(1)}),
        (ALIN, DGN0, {'out': ALPH.format(0), 'in': ALPH.format(0)}),
        (ALIN, DGN1, {'out': ALPH.format(1), 'in': ALPH.format(1)}),
    ])

    return G
