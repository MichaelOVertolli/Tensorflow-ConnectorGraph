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
BRGF = 'res_bridge_00'
DSC0 = 'res_rev_00'
DSC1 = 'res_rev_01'
DSC2 = 'res_rev_02'
BRGD = 'res_bridge_01'
DGN0 = 'res_gen_pair_03'
DGN1 = 'res_gen_pair_04'
DGN2 = 'res_gen_pair_05'
BRGU = 'res_bridge_02'
REST = 'res_train'
NCNT = 'mad_nconcat_00'
NSPL = 'mad_nsplit_00'
LSSG = 'res_loss_set_00'
LSSD = 'res_loss_set_01'
LSSN = 'res_loss_set_02'


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

#Variables
VARS = '/trainable_variables'
VARIABLES = '/variables'


def build(config):
    G = nx.DiGraph(
        train_scope='res_train',
        greyscale=False,
        # frozen=['res_gen_pair', 'res_rev'],
        block_index=0,
        last_index=5,
        counts={
            'res_bridge': 3,
            'res_rev': 3,
            'res_gen_pair': 6,
            'res_loss_set': 3},
        loss_tensors={
            'G': LSSG+OUTP,
            'D': LSSD+OUTP,
            'N': LSSN+OUTP,},
        train_sets={
            'G': [GNF0, GNF1, GNF2, BRGF],
            'D': [BRGD, DSC2, DSC1, DSC0, DGN0, DGN1, DGN2, BRGU],},
        img_pairs=[
            ('G', BRGF+OUTP),
            ('A_G', NSPL+OUTN.format(2)),
            ('A_N', NSPL+OUTN.format(1)),
            ('A_D', NSPL+OUTN.format(0)),],
        saver_pairs=[
            (GNF0, GNF0+VARIABLES),
            (GNF1, GNF1+VARIABLES),
            (GNF2, GNF2+VARIABLES),
            (BRGF, BRGF+VARIABLES),
            (BRGD, BRGD+VARIABLES),
            (DSC0, DSC0+VARIABLES),
            (DSC1, DSC1+VARIABLES),
            (DSC2, DSC2+VARIABLES),
            (DGN0, DGN0+VARIABLES),
            (DGN1, DGN1+VARIABLES),
            (DGN2, DGN2+VARIABLES),
            (BRGU, BRGU+VARIABLES),],
        gen_tensor=BRGF+OUTP,
        gen_input=GNF0+INPT,
        rev_input=None,
        noise_input=NCNT+INPN.format(1),
        data_inputs=[
            NCNT+INPN.format(0),
            LSSD+O_IN,
            LSSN+O_IN],
        alpha_tensor=None,
        gen_outputs={
            'G': BRGF+OUTP,},
        a_output=NSPL+OUTN.format(0),
        an_output=NSPL+OUTN.format(1),
        growth_types=None,
    )
    
    # Input SubGraphs    
    G.add_nodes_from([
        (GNF0, {'inputs': [INPT]}),
        (NCNT, {'inputs': [INPN.format(0), INPN.format(1)]}),
        (LSSD, {'inputs': [O_IN]}),
        (LSSN, {'inputs': [O_IN]}),
    ])
    # Output SubGraphs
    G.add_nodes_from([
        (BRGF, {'outputs': [OUTP]}),
        (LSSG, {'outputs': [OUTP]}),
        (LSSD, {'outputs': [OUTP]}),
        (LSSN, {'outputs': [OUTP]}),
    ])
    
    # SubGraphs with config mods
    G.add_nodes_from([
        (BRGF, {'config': ['to_image', 'clone']}),
        (BRGU, {'config': ['to_image']}),
        (GNF0, {'config': ['block0', 'clone', 'base']}),
        (GNF1, {'config': ['block1', 'clone']}),
        (GNF2, {'config': ['block2', 'clone']}),
        (DSC0, {'config': ['block0', 'base']}),
        (DSC1, {'config': ['block1']}),
        (DSC2, {'config': ['block2']}),
        (DGN0, {'config': ['block0', 'base']}),
        (DGN1, {'config': ['block1']}),
        (DGN2, {'config': ['block2']}),
        (NCNT, {'config': ['count3']}),
        (NSPL, {'config': ['count3']}),
        (LSSN, {'config': ['mse']}),
    ])
    
    # Remaining inner SubGraphs
    G.add_nodes_from([BRGD])

    G.add_edges_from([
        # Reverse path
        (GNF0, GNF1, {'out': OUTP, 'in': INPT}),
        (GNF1, GNF2, {'out': OUTP, 'in': INPT}),
        (GNF2, BRGF, {'out': OUTP, 'in': INPT}),
        # Main path
        (BRGF, NCNT, {'out': OUTP, 'in': INPN.format(2)}),
        (NCNT, BRGD, {'out': OUTP, 'in': INPT}),
        (BRGD, DSC2, {'out': OUTP, 'in': INPT}),
        (DSC2, DSC1, {'out': OUTP, 'in': INPT}),
        (DSC1, DSC0, {'out': OUTP, 'in': INPT}),
        (DSC0, DGN0, {'out': OUTP, 'in': INPT}),
        (DGN0, DGN1, {'out': OUTP, 'in': INPT}),
        (DGN1, DGN2, {'out': OUTP, 'in': INPT}),
        (DGN2, BRGU, {'out': OUTP, 'in': INPT}),
        (BRGU, NSPL, {'out': OUTP, 'in': INPT}),
        (BRGF, LSSG, {'out': OUTP, 'in': O_IN}),
        (NSPL, LSSG, {'out': OUTN.format(2), 'in': A_IN}),
        (NSPL, LSSN, {'out': OUTN.format(1), 'in': A_IN}),
        (NSPL, LSSD, {'out': OUTN.format(0), 'in': A_IN}),        
    ])

    return G
