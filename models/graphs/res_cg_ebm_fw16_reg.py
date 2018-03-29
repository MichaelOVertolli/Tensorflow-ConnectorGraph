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
GNF0 = 'res_gen_pair_00'
GNF1 = 'res_gen_pair_01'
BRGF = 'res_bridge_00'
DSC0 = 'res_rev_00'
DSC1 = 'res_rev_01'
BRGD = 'res_bridge_01'
DGN0 = 'res_gen_pair_02'
DGN1 = 'res_gen_pair_03'
BRGU = 'res_bridge_02'
REST = 'res_train'
CNCT = 'cqs_concat_00'
SPLT = 'cqs_split_00'
LSSG = 'res_loss_set_00'
LSSD = 'res_loss_set_01'
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
        block_index=1,
        last_index=4,
        counts={
            'res_bridge': 3,
            'res_rev': 2,
            'res_gen_pair': 4,
            'res_loss_set': 2},
        loss_tensors={
            'G': LSSG+OUTP,
            'D': LSSD+OUTP,},
        train_sets={
            'G': [GNF0, GNF1, BRGF],
            'D': [BRGD, DSC1, DSC0, DGN0, DGN1, BRGU],},
        img_pairs=[
            ('G', BRGF+OUTP),
            ('A_G', SPLT+GOUT),
            ('A_D', SPLT+DOUT),],
        saver_pairs=[
            (GNF0, GNF0+VARIABLES),
            (GNF1, GNF1+VARIABLES),
            (BRGF, BRGF+VARIABLES),
            (BRGD, BRGD+VARIABLES),
            (DSC1, DSC1+VARIABLES),
            (DSC0, DSC0+VARIABLES),
            (DGN0, DGN0+VARIABLES),
            (DGN1, DGN1+VARIABLES),
            (BRGU, BRGU+VARIABLES),],
        gen_tensor=BRGF+OUTP,
        gen_input=GNF0+INPT,
        rev_input=None,
        data_inputs=[
            CNCT+D_IN,
            LSSD+O_IN,],
        alpha_tensor=ALIN+ALPH,
        gen_outputs={
            'G': BRGF+OUTP,},
        a_output=SPLT+DOUT,
        growth_types={
            BRGF: {
                'name': 'res_gen_pair_{:02}',
                'in': INPT,
                'out': OUTP,
                'config': 'block{}_clone',
                'train': 'G',
                'rev': False,
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
        (GNF0, {'inputs': [INPT]}),
        (CNCT, {'inputs': [D_IN]}),
        (LSSD, {'inputs': [O_IN]}),
        (ALIN, {'inputs': alpha_inputs}),
    ])
    # Output SubGraphs
    G.add_nodes_from([
        (BRGF, {'outputs': [OUTP]}),
        (LSSG, {'outputs': [OUTP]}),
        (LSSD, {'outputs': [OUTP]}),
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
        (GNF0, {'config': ['block0', 'clone', 'base']}),
        (GNF1, {'config': ['block1', 'clone']}),
        (DSC1, {'config': ['block1']}),
        (DSC0, {'config': ['block0', 'base']}),
        (DGN0, {'config': ['block0', 'base']}),
        (DGN1, {'config': ['block1']}),
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
        # Main path
        # (GENF, BRGF, {'out': [OUTP, OUT2], 'in': [INPT, INP2]}),
        (GNF0, GNF1, {'out': OUTP, 'in': INPT}),
        (GNF1, BRGF, {'out': OUTP, 'in': INPT, 'mod': BRGF}),
        (BRGF, CNCT, {'out': OUTP, 'in': G_IN}),
        (CNCT, BRGD, {'out': OUTP, 'in': INPT}),
        (BRGD, DSC1, {'out': OUTP, 'in': INPT, 'mod': BRGD}),
        (DSC1, DSC0, {'out': OUTP, 'in': INPT}),
        (DSC0, DGN0, {'out': OUTP, 'in': INPT}),
        (DGN0, DGN1, {'out': OUTP, 'in': INPT}),
        (DGN1, BRGU, {'out': OUTP, 'in': INPT, 'mod': BRGU}),
        (BRGU, SPLT, {'out': OUTP, 'in': INPT}),
        (BRGF, LSSG, {'out': OUTP, 'in': O_IN}),
        (SPLT, LSSG, {'out': GOUT, 'in': A_IN}),
        (SPLT, LSSD, {'out': DOUT, 'in': A_IN}),
        # Alphas
        (ALIN, GNF0, {'out': ALPH.format(0), 'in': ALPH.format(0)}),
        (ALIN, GNF1, {'out': ALPH.format(1), 'in': ALPH.format(1)}),
        (ALIN, DSC0, {'out': ALPH.format(0), 'in': ALPH.format(0)}),
        (ALIN, DSC1, {'out': ALPH.format(1), 'in': ALPH.format(1)}),
        (ALIN, DGN0, {'out': ALPH.format(0), 'in': ALPH.format(0)}),
        (ALIN, DGN1, {'out': ALPH.format(1), 'in': ALPH.format(1)}),
    ])

    return G
