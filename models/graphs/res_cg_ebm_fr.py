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
GENF = 'res_gen_pair_00'
BRGF = 'res_bridge_00'
GENR = 'res_rev_00'
BRGR = 'res_bridge_01'
DISC = 'res_rev_01'
BRGD = 'res_bridge_02'
DGEN = 'res_gen_pair_01'
BRGU = 'res_bridge_03'
REST = 'res_train'
CNCT = 'cqs_concat_00'
SPLT = 'cqs_split_00'
LSSG = 'cqs_loss_set_00'
LSSD = 'cqs_loss_set_01'
LSSR = 'cqs_loss_set_02'
ALIN = 'res_alphas_0'

#Inputs
INPT = '/input:0'
INP2 = '/input2:0'
G_IN = '/gen_input:0'
D_IN = '/data_input:0'
M_IN = '/mix_input:0'
O_IN = '/orig_input:0'
A_IN = '/autoencoded_input:0'

#Outputs
OUTP = '/output:0'
OUT2 = '/output2:0'
GOUT = '/gen_output:0'
DOUT = '/data_output:0'
MOUT = '/mix_output:0'

#Alphas
ALPH = '/alpha{}:0'

#Variables
VARS = '/trainable_variables'
VARIABLES = '/variables'


def build(config):
    G = nx.DiGraph(
        train_scope='res_train',
        greyscale=False,
        # frozen=['res_gen_pair', 'res_rev'],
        block_index=0,
        last_index=2,
        counts={
            'res_bridge': 4,
            'res_rev': 2,
            'res_gen_pair': 2,
            'cqs_loss_set': 3},
        loss_tensors={
            'G': LSSG+OUTP,
            'R': LSSR+OUTP,
            'D': LSSD+OUTP,},
        train_sets={
            'G': [GENF, BRGF],
            'R': [GENR, BRGR],
            'D': [DISC, DGEN, BRGD, BRGU],},
        img_pairs=[
            ('G', BRGF+OUTP),
            ('R', BRGF+OUT2),
            ('A_G', SPLT+GOUT),
            ('A_D', SPLT+DOUT),],
        saver_pairs=[
            (BRGR, BRGR+VARIABLES),
            (GENR, GENR+VARIABLES),
            (GENF, GENF+VARIABLES),
            (BRGF, BRGF+VARIABLES),
            (BRGD, BRGD+VARIABLES),
            (DISC, DISC+VARIABLES),
            (DGEN, DGEN+VARIABLES),
            (BRGU, BRGU+VARIABLES),],
        gen_tensor=BRGF+OUTP,
        gen_input=GENF+INPT,
        rev_input=BRGR+INPT,
        data_inputs=[
            CNCT+D_IN,
            LSSD+O_IN,],
        alpha_tensor=ALIN+ALPH,
        gen_outputs={
            'G': BRGF+OUTP,
            'R': BRGF+OUT2,},
        a_output=SPLT+DOUT,
        growth_types={
            BRGF: {
                'name': 'res_gen_pair_{:02}',
                'in': [INPT, INP2],
                'out': [OUTP, OUT2],
                'config': 'block{}_clone',
                'train': 'G',
                'rev': False,
            },
            BRGR: {
                'name': 'res_rev_{:02}',
                'in': INPT,
                'out': OUTP,
                'config': 'block{}',
                'train': 'R',
                'rev': True,
            },
            BRGD: {
                'name': 'res_rev_0{:02}',
                'in': INPT,
                'out': OUTP,
                'config': 'block{}',
                'train': 'D',
                'rev': True,
            },
            BRGU: {
                'name': 'res_gen_pair_{:02}',
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
        (GENF, {'inputs': [INPT]}),
        (GENR, {'inputs': [INPT]}),
        (CNCT, {'inputs': [D_IN]}),
        (LSSD, {'inputs': [O_IN]}),
        (ALIN, {'inputs': alpha_inputs}),
    ])
    # Output SubGraphs
    G.add_nodes_from([
        (BRGF, {'outputs': [OUTP, OUT2]}),
        (GENR, {'outputs': [OUTP]}),
        (LSSG, {'outputs': [OUTP]}),
        (LSSD, {'outputs': [OUTP]}),
        (LSSR, {'outputs': [OUTP]}),
    ])
    # Image outputs
    # G.add_nodes_from([
    #     (BRGF, {'img': [OUTP, OUT2]}),
    #     (BRGU, {'img': [OUTP]}),
    # ])
    # SubGraphs with config mods
    G.add_nodes_from([
        (BRGF, {'config': ['to_image', 'clone']}),
        (BRGU, {'config': ['to_image']}),
        (GENF, {'config': ['block0', 'clone', 'base']}),
        (GENR, {'config': ['block0', 'base']}),
        (DISC, {'config': ['block0', 'base']}),
        (DGEN, {'config': ['block0', 'base']}),
    ])
    # Train sets
    # G.add_nodes_from([
    #     (GENF, {'train': ['G']}),
    #     (BRGF, {'train': ['G']}),
    #     (BRGR, {'train': ['R']}),
    #     (GENR, {'train': ['R']}),
    #     (BRGD, {'train': ['D']}),
    #     (DISC, {'train': ['D']}),
    #     (DGEN, {'train': ['D']}),
    #     (BRGU, {'train': ['D']}),
    # ])
    # Loss sets
    # G.add_nodes_from([
    #     (LSSG, {'loss': ['G']}),
    #     (LSSD, {'loss': ['D']}),
    #     (LSSR, {'loss': ['R']}),
    # ])
    # Remaining inner SubGraphs
    G.add_nodes_from([SPLT])

    G.add_edges_from([
        # Reverse path
        (BRGR, GENR, {'out': OUTP, 'in': INPT, 'mod': BRGR}),
        (GENR, GENF, {'out': OUTP, 'in': INP2}),
        (GENF, BRGF, {'out': [OUTP, OUT2], 'in': [INPT, INP2], 'mod': BRGF}),
        (BRGF, LSSR, {'out': [OUTP, OUT2], 'in': [O_IN, A_IN]}),
        # Main path
        # (GENF, BRGF, {'out': [OUTP, OUT2], 'in': [INPT, INP2]}),
        (BRGF, CNCT, {'out': OUTP, 'in': G_IN}),
        (CNCT, BRGD, {'out': OUTP, 'in': INPT}),
        (BRGD, DISC, {'out': OUTP, 'in': INPT, 'mod': BRGD}),
        (DISC, DGEN, {'out': OUTP, 'in': INPT}),
        (DGEN, BRGU, {'out': OUTP, 'in': INPT, 'mod': BRGU}),
        (BRGU, SPLT, {'out': OUTP, 'in': INPT}),
        (BRGF, LSSG, {'out': OUTP, 'in': O_IN}),
        (SPLT, LSSG, {'out': GOUT, 'in': A_IN}),
        (SPLT, LSSD, {'out': DOUT, 'in': A_IN}),
        # Alphas
        (ALIN, GENF, {'out': ALPH.format(0), 'in': ALPH.format(0)}),
        (ALIN, GENR, {'out': ALPH.format(0), 'in': ALPH.format(0)}),
        (ALIN, DISC, {'out': ALPH.format(0), 'in': ALPH.format(0)}),
        (ALIN, DGEN, {'out': ALPH.format(0), 'in': ALPH.format(0)}),
    ])

    return G
