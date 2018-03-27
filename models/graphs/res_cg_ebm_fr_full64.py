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
GNF2 = 'res_gen_pair_02'
GNF3 = 'res_gen_pair_03'
GNF4 = 'res_gen_pair_04'
BRGF = 'res_bridge_00'
GNR0 = 'res_rev_00'
GNR1 = 'res_rev_01'
GNR2 = 'res_rev_02'
GNR3 = 'res_rev_03'
GNR4 = 'res_rev_04'
BRGR = 'res_bridge_01'
DSC0 = 'res_rev_05'
DSC1 = 'res_rev_06'
DSC2 = 'res_rev_07'
DSC3 = 'res_rev_08'
DSC4 = 'res_rev_09'
BRGD = 'res_bridge_02'
DGN0 = 'res_gen_pair_05'
DGN1 = 'res_gen_pair_06'
DGN2 = 'res_gen_pair_07'
DGN3 = 'res_gen_pair_08'
DGN4 = 'res_gen_pair_09'
BRGU = 'res_bridge_03'
REST = 'res_train'
CNCT = 'cqs_concat_00'
SPLT = 'cqs_split_00'
LSSG = 'cqs_loss_set_00'
LSSD = 'cqs_loss_set_01'
LSSR = 'cqs_loss_set_02'


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

#Variables
VARS = '/trainable_variables'
VARIABLES = '/variables'


def build(config):
    G = nx.DiGraph(
        train_scope='res_train',
        greyscale=False,
        # frozen=['res_gen_pair', 'res_rev'],
        block_index=0,
        last_index=0,
        counts={
            'res_bridge': 3,
            'res_rev': 9,
            'res_gen_pair': 9,
            'cqs_loss_set': 3},
        loss_tensors={
            'G': LSSG+OUTP,
            'R': LSSR+OUTP,
            'D': LSSD+OUTP,},
        train_sets={
            'G': [GNF0, GNF1, GNF2, GNF3, BRGF],
            'R': [BRGR, GNR4, GNR3, GNR2, GNR0],
            'D': [BRGD, DSC3, DSC2, DSC1, DSC0, DGN0, DGN1, DGN2, DGN3, BRGU],},
        img_pairs=[
            ('G', BRGF+OUTP),
            ('R', BRGF+OUT2),
            ('A_G', SPLT+GOUT),
            ('A_D', SPLT+DOUT),],
        saver_pairs=[
            (BRGR, BRGR+VARIABLES),
            (GNR0, GNR0+VARIABLES),
            (GNR1, GNR1+VARIABLES),
            (GNR2, GNR2+VARIABLES),
            (GNR3, GNR3+VARIABLES),
            (GNF0, GNF0+VARIABLES),
            (GNF1, GNF1+VARIABLES),
            (GNF2, GNF2+VARIABLES),
            (GNF3, GNF3+VARIABLES),
            (BRGF, BRGF+VARIABLES),
            (BRGD, BRGD+VARIABLES),
            (DSC0, DSC0+VARIABLES),
            (DSC1, DSC1+VARIABLES),
            (DSC2, DSC2+VARIABLES),
            (DSC3, DSC3+VARIABLES),
            (DGN0, DGN0+VARIABLES),
            (DGN1, DGN1+VARIABLES),
            (DGN2, DGN2+VARIABLES),
            (DGN3, DGN3+VARIABLES),
            (BRGU, BRGU+VARIABLES),],
        gen_tensor=BRGF+OUTP,
        gen_input=GNF0+INPT,
        rev_input=BRGR+INPT,
        data_inputs=[
            CNCT+D_IN,
            LSSD+O_IN,],
        alpha_tensor=None,
        gen_outputs={
            'G': BRGF+OUTP,
            'R': BRGF+OUT2,},
        a_output=SPLT+DOUT,
        growth_types=None
    )
    
    # Input SubGraphs    
    G.add_nodes_from([
        (GNF0, {'inputs': [INPT]}),
        (BRGR, {'inputs': [INPT]}),
        (CNCT, {'inputs': [D_IN]}),
        (LSSD, {'inputs': [O_IN]}),
    ])
    # Output SubGraphs
    G.add_nodes_from([
        (BRGF, {'outputs': [OUTP, OUT2]}),
        (GNR0, {'outputs': [OUTP]}),
        (LSSG, {'outputs': [OUTP]}),
        (LSSD, {'outputs': [OUTP]}),
        (LSSR, {'outputs': [OUTP]}),
    ])
    
    # SubGraphs with config mods
    G.add_nodes_from([
        (BRGF, {'config': ['to_image', 'clone']}),
        (BRGU, {'config': ['to_image']}),
        (GNF0, {'config': ['block0', 'clone', 'base']}),
        (GNF1, {'config': ['block1', 'clone']}),
        (GNF2, {'config': ['block2', 'clone']}),
        (GNF3, {'config': ['block3', 'clone']}),
        (GNR0, {'config': ['block0', 'base']}),
        (GNR1, {'config': ['block1']}),
        (GNR2, {'config': ['block2']}),
        (GNR3, {'config': ['block3']}),
        (DSC0, {'config': ['block0', 'base']}),
        (DSC1, {'config': ['block1']}),
        (DSC2, {'config': ['block2']}),
        (DSC3, {'config': ['block3']}),
        (DGN0, {'config': ['block0', 'base']}),
        (DGN1, {'config': ['block1']}),
        (DGN2, {'config': ['block2']}),
        (DGN3, {'config': ['block3']}),
    ])
    
    # Remaining inner SubGraphs
    G.add_nodes_from([SPLT, BRGD])

    G.add_edges_from([
        # Reverse path
        (BRGR, GNR3, {'out': OUTP, 'in': INPT}),
        (GNR3, GNR2, {'out': OUTP, 'in': INPT}),
        (GNR2, GNR1, {'out': OUTP, 'in': INPT}),
        (GNR1, GNR0, {'out': OUTP, 'in': INPT}),
        (GNR0, GNF0, {'out': OUTP, 'in': INP2}),
        (GNF0, GNF1, {'out': [OUTP, OUT2], 'in': [INPT, INP2]}),
        (GNF1, GNF2, {'out': [OUTP, OUT2], 'in': [INPT, INP2]}),
        (GNF2, GNF3, {'out': [OUTP, OUT2], 'in': [INPT, INP2]}),
        (GNF3, BRGF, {'out': [OUTP, OUT2], 'in': [INPT, INP2]}),
        (BRGF, LSSR, {'out': [OUTP, OUT2], 'in': [O_IN, A_IN]}),
        # Main path
        (BRGF, CNCT, {'out': OUTP, 'in': G_IN}),
        (CNCT, BRGD, {'out': OUTP, 'in': INPT}),
        (BRGD, DSC3, {'out': OUTP, 'in': INPT}),
        (DSC3, DSC2, {'out': OUTP, 'in': INPT}),
        (DSC2, DSC1, {'out': OUTP, 'in': INPT}),
        (DSC1, DSC0, {'out': OUTP, 'in': INPT}),
        (DSC0, DGN0, {'out': OUTP, 'in': INPT}),
        (DGN0, DGN1, {'out': OUTP, 'in': INPT}),
        (DGN1, DGN2, {'out': OUTP, 'in': INPT}),
        (DGN2, DGN3, {'out': OUTP, 'in': INPT}),
        (DGN3, BRGU, {'out': OUTP, 'in': INPT}),
        (BRGU, SPLT, {'out': OUTP, 'in': INPT}),
        (BRGF, LSSG, {'out': OUTP, 'in': O_IN}),
        (SPLT, LSSG, {'out': GOUT, 'in': A_IN}),
        (SPLT, LSSD, {'out': DOUT, 'in': A_IN}),
    ])

    return G
