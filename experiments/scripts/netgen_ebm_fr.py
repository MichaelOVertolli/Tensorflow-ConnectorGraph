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

from netgen import *
import networkx as nx

def test():
    n = NetGen('res_cg_ebm_fr',
               'scaled_began_gmsm_b16_z256_sz32_h512_g0.7_elu_pconv_wxav_alphas',
               'grass',
               [{'dir': 'base_block', 'data': 'grass8'},
                {'dir': 'block1', 'data': 'grass16'},
                {'dir': 'block2', 'data': 'grass32'},])
    #n.t_config.max_step = 500

    #n.add_subgraphs(1)
    #nx.drawing.nx_agraph.write_dot(n.G, './test.dot')
    #c = n.convert_cg({})

    n.run()

    return n
