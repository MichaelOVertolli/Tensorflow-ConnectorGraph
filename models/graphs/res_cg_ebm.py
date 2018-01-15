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


#Models
GENR = 'res_generator_0'
DISC = 'res_discriminator_0'
DGEN = 'res_generator_1'
REST = 'res_train'
CNCT = 'cqs_concat_0'
SPLT = 'cqs_split_0'
LSSG = 'cqs_loss_set_0'
LSSD = 'cqs_loss_set_1'

#Inputs
INPT = '/input:0'
G_IN = '/gen_input:0'
D_IN = '/data_input:0'
M_IN = '/mix_input:0'
O_IN = '/orig_input:0'
A_IN = '/autoencoded_input:0'

#Outputs
OUTP = '/output:0'
GOUT = '/gen_output:0'
DOUT = '/data_output:0'
MOUT = '/mix_output:0'

#Alphas
ALPH = '/alpha{}:0'

#Variables
VARS = '/trainable_variables'
VARIABLES = '/variables'


def build():
    G = nx.DiGraph()
    #Input SubGraphs
    G.add_nodes_from([
        (GENR, {'inputs': [INPT]}),
        (CNCT, {'inputs': [D_IN]}),
        (LSSD, {'inputs': [O_IN]}),
    ])
    #Output SubGraphs
    G.add_nodes_from([
        (LSSG, {'outputs': [OUTP]}),
        (LSSD, {'outputs': [OUTP]}),
    ])
    #Inner SubGraphs
    G.add_nodes_from([DISC, DGEN, SPLT])

    G.add_edges_from([
        (GENR, CNCT, {'out': OUTP, 'in': G_IN}),
        (CNCT, DISC, {'out': OUTP, 'in': INPT}),
        (DISC, DGEN, {'out': OUTP, 'in': INPT}),
        (DGEN, SPLT, {'out': OUTP, 'in': INPT}),
        (GENR, LSSG, {'out': OUTP, 'in': O_IN}),
        (SPLT, LSSG, {'out': GOUT, 'in': A_IN}),
        (SPLT, LSSD, {'out': DOUT, 'in': A_IN}),
    ])

    return G
