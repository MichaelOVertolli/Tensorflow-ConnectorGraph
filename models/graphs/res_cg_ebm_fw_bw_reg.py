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
DISC = 'res_rev_00'
BRGD = 'res_bridge_01'
DGEN = 'res_gen_pair_01'
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
        greyscale=True,
        # frozen=['res_gen_pair', 'res_rev'],
        block_index=0,
        last_index=5,
        counts={
            'res_bridge': 3,
            'res_rev': 1,
            'res_gen_pair': 2,
            'cqs_loss_set': 3},
        loss_tensors={
            'G': LSSG+OUTP,
            'D': LSSD+OUTP,},
        train_sets={
            'G': [GENF, BRGF],
            'D': [DISC, DGEN, BRGD, BRGU],},
        img_pairs=[
            ('G', BRGF+OUTP),
            ('A_G', SPLT+GOUT),
            ('A_D', SPLT+DOUT),],
        saver_pairs=[
            (GENF, GENF+VARIABLES),
            (BRGF, BRGF+VARIABLES),
            (BRGD, BRGD+VARIABLES),
            (DISC, DISC+VARIABLES),
            (DGEN, DGEN+VARIABLES),
            (BRGU, BRGU+VARIABLES),],
        gen_tensor=BRGF+OUTP,
        gen_input=GENF+INPT,
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
        (GENF, {'inputs': [INPT]}),
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
    # SubGraphs with config mods
    G.add_nodes_from([
        (BRGF, {'config': ['to_image', 'clone', 'b+w']}),
        (BRGU, {'config': ['to_image', 'b+w']}),
        (BRGD, {'config': ['b+w']}),
        (GENF, {'config': ['block0', 'clone', 'base']}),
        (DISC, {'config': ['block0', 'base']}),
        (DGEN, {'config': ['block0', 'base']}),
        (LSSG, {'config': ['b+w']}),
        (LSSD, {'config': ['b+w']}),
    ])

    # Remaining inner SubGraphs
    G.add_nodes_from([SPLT, BRGD])

    G.add_edges_from([
        # Main path
        # (GENF, BRGF, {'out': [OUTP, OUT2], 'in': [INPT, INP2]}),
        (GENF, BRGF, {'out': OUTP, 'in': INPT, 'mod': BRGF}),
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
        (ALIN, DISC, {'out': ALPH.format(0), 'in': ALPH.format(0)}),
        (ALIN, DGEN, {'out': ALPH.format(0), 'in': ALPH.format(0)}),
    ])

    return G
