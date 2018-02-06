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
GNF0 = 'res_gen_pair_00000000'
GNF1 = 'res_gen_pair_00000020'
GNF2 = 'res_gen_pair_00000030'
BRF0 = 'res_bridge_00'
BRF1 = 'res_bridge_01'
GNR0 = 'res_rev_00000000'
GNR1 = 'res_rev_00000020'
GNR2 = 'res_rev_00000030'
BRR0 = 'res_bridge_10'
BRR1 = 'res_bridge_11'
DSC0 = 'res_rev_00'
DSC1 = 'res_rev_01'
BRGD = 'res_bridge_20'
DGN0 = 'res_gen_pair_10'
DGN1 = 'res_gen_pair_11'
BRGU = 'res_bridge_30'
REST = 'res_train'
NCN0 = 'mad_nconcat_0'
NCN1 = 'mad_nconcat_1'
NSP0 = 'mad_nsplit_0'
NSP1 = 'mad_nsplit_1'
LSG0 = 'cqs_loss_set_00'
LSG1 = 'cqs_loss_set_01'
LSR0 = 'cqs_loss_set_10'
LSR1 = 'cqs_loss_set_11'
LSSD = 'cqs_loss_set_20'
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
        block_index=1,
        breadth_count=2,
        subnets={
            'generic': [LSSD, LSSU],
            'leaf_nodes': [[LSG0, LSR0], [LSG1, LSR1]],},
        loss_tensors={
            'G': [LSG0+OUTP, LSG1+OUTP],
            'R': [LSR0+OUTP, LSR1+OUTP],
            'D': LSSD+OUTP,
            'U': [LSSU+OUTN.format(i) for i in range(BCOUNT)],},
        train_sets={
            'G': [[GNF0, GNF1, BRF0], [GNF0, GNF2, BRF1]],
            'R': [[BRR0, GNR1, GNR0], [BRR1, GNR2, GNR0]],
            'D': [DSC0, DSC1, DGN0, DGN1, BRGD, BRGU],},
        img_pairs=[
            ('G1', BRF0+OUTP),
            ('G2', BRF1+OUTP),
            ('R1', BRF0+OUT2),
            ('R2', BRF1+OUT2),
            ('A_G1', NSP0+OUTN.format(1)),
            ('A_G2', NSP0+OUTN.format(2)),
            ('A_D', NSP0+OUTN.format(0)),],
        saver_pairs=[
            (BRR0, BRR0+VARIABLES),
            (BRR1, BRR1+VARIABLES),
            (GNR0, GNR0+VARIABLES),
            (GNR1, GNR1+VARIABLES),
            (GNR2, GNR2+VARIABLES),
            (GNF0, GNF0+VARIABLES),
            (GNF1, GNF1+VARIABLES),
            (GNF2, GNF2+VARIABLES),
            (BRF0, BRF0+VARIABLES),
            (BRF1, BRF1+VARIABLES),
            (BRGD, BRGD+VARIABLES),
            (DSC0, DSC0+VARIABLES),
            (DSC1, DSC1+VARIABLES),
            (DGN0, DGN0+VARIABLES),
            (DGN1, DGN1+VARIABLES),
            (BRGU, BRGU+VARIABLES),],
        gen_tensor=[BRF0+OUTP, BRF1+OUTP],
        gen_input=GNR0+INPT,
        rev_inputs=[BRR0+INPT, BRR1+INPT],
        data_inputs=[
            NCN0+INPN.format(0),
            LSSD+O_IN,],
        alpha_tensor=ALIN+ALPH,
        gen_outputs={
            'G': [BRF0+OUTP, BRF1+OUTP],
            'R': [BRF0+OUT2, BRF1+OUT2]},
        a_output=NSP0+OUTN.format(0),
        growth_groups={
            'res_gen_pair': {
                'new_subgraph': 'res_gen_pair_{}00', # add leading zeros up to 6
                'in': [INPT, INP2],
                'out': [OUTP, OUT2],
                'config': 'block{}_clone',
                'train': 'G',
                'count': 2,
            },
            'res_rev': {
                'new_subgraph': 'res_rev_{}00',
                'in': INPT,
                'out': OUTP,
                'config': 'block{}',
                'train': 'R',
                'count': 2,
            },
        },
        growth_types={
            BRGD: {
                'new_subgraph': 'res_rev_0{}',
                'in': INPT,
                'out': OUTP,
                'config': 'block{}',
                'train': 'D',
            },
            BRGU: {
                'new_subgraph': 'res_gen_pair_1{}',
                'in': INPT,
                'out': OUTP,
                'config': 'block{}',
                'train': 'D',
            },
        },
    )
    
    alpha_inputs = []
    for i in range(config.repeat_num-1):
        alpha_inputs.append(ALPH.format(i))
    
    # Input SubGraphs    
    G.add_nodes_from([
        (GNF0, {'inputs': [INPT]}),
        (BRR0, {'inputs': [INPT]}),
        (BRR1, {'inputs': [INPT]}),
        (NCN0, {'inputs': [OUTN]}),
        (LSSD, {'inputs': [O_IN]}),
        (ALIN, {'inputs': alpha_inputs}),
    ])
    # Output SubGraphs
    G.add_nodes_from([
        (BRF0, {'outputs': [OUTP, OUT2]}),
        (BRF1, {'outputs': [OUTP, OUT2]}),
        (GNR0, {'outputs': [OUTP]}),
        (LSG0, {'outputs': [OUTP]}),
        (LSG1, {'outputs': [OUTP]}),
        (LSR0, {'outputs': [OUTP]}),
        (LSR1, {'outputs': [OUTP]}),
        (LSSD, {'outputs': [OUTP]}),
        (LSSU, {'outputs': [OUTN.format(i) for i in range(G.graph['breadth_count'])]}),
    ])
    # SubGraphs with config mods
    G.add_nodes_from([
        (BRF0, {'config': ['to_image', 'clone']}),
        (BRF1, {'config': ['to_image', 'clone']}),
        (BRGU, {'config': ['to_image']}),
        (GNF0, {'config': ['block0', 'clone']}),
        (GNF1, {'config': ['block1', 'clone']}),
        (GNF2, {'config': ['block1', 'clone']}),
        (GNR0, {'config': ['block0']}),
        (GNR1, {'config': ['block1']}),
        (GNR2, {'config': ['block1']}),
        (DSC0, {'config': ['block0']}),
        (DSC1, {'config': ['block1']}),
        (DGN0, {'config': ['block0']}),
        (DGN1, {'config': ['block1']}),
        (NCN0, {'config': ['count{}'.format(G.graph['breadth_count']+1)]}), # breadth_count + real_input
        (NCN1, {'config': ['count{}'.format(2)]}),
        (NSP0, {'config': ['count{}'.format(G.graph['breadth_count']+1)]}),
        (NSP1, {'config': ['count{}'.format(2)]}),
        (LSSU, {'config': ['count{}'.format(G.graph['breadth_count']+1)]}), 
    ])
    # Remaining inner SubGraphs
    # None

    G.add_edges_from([
        # Reverse path
        (NSP1, BRR0, {'out': OUTN.format(0), 'in': INPT}),
        (BRR0, GNR1, {'out': OUTP, 'in': INPT, 'mod': BRR0}),
        (GNR1, NCN1, {'out': OUTP, 'in': INPN.format(0), 'del': True}),
        (NSP1, BRR1, {'out': OUTN.format(1), 'in': INPT}),
        (BRR1, GNR2, {'out': OUTP, 'in': INPT, 'mod': BRR1}),
        (GNR2, NCN1, {'out': OUTP, 'in': INPN.format(1), 'del': True}),
        (NCN1, GNR0, {'out': OUTP, 'in': INPT}),
        (GNR0, GNF0, {'out': OUTP, 'in': INP2}),
        (GNF0, GNF1, {'out': [OUTP, OUT2], 'in': [INPT, INP2]}),
        (GNF1, BRF0, {'out': [OUTP, OUT2], 'in': [INPT, INP2], 'mod': BRF0}),
        (GNF0, GNF2, {'out': [OUTP, OUT2], 'in': [INPT, INP2]}),
        (BRF0, LSR0, {'out': [OUTP, OUT2], 'in': [O_IN, A_IN]}),
        (GNF2, BRF1, {'out': [OUTP, OUT2], 'in': [INPT, INP2], 'mod': BRF1}),
        (BRF1, LSR1, {'out': [OUTP, OUT2], 'in': [O_IN, A_IN]}),
        # Main path
        # (GENF, BRGF, {'out': [OUTP, OUT2], 'in': [INPT, INP2]}),
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
        (ALIN, GNR0, {'out': ALPH.format(0), 'in': ALPH.format(0)}),
        (ALIN, GNR1, {'out': ALPH.format(1), 'in': ALPH.format(1)}),
        (ALIN, GNR2, {'out': ALPH.format(1), 'in': ALPH.format(1)}),
        (ALIN, DSC0, {'out': ALPH.format(0), 'in': ALPH.format(0)}),
        (ALIN, DSC1, {'out': ALPH.format(1), 'in': ALPH.format(1)}),
        (ALIN, DGN0, {'out': ALPH.format(0), 'in': ALPH.format(0)}),
        (ALIN, DGN1, {'out': ALPH.format(1), 'in': ALPH.format(1)}),
    ])

    return G
