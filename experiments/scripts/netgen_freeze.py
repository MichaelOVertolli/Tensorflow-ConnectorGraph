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
from tessellater import *

def test():
    n = NetGen('res_cg_ebm_fr',
               'scaled_began_gmsm_b16_z256_sz32_h256_g0.7_elu_pconv_wxav_alphas',
               'grass',
               [{'dir': 'base_block', 'data': 'grass8'},
                {'dir': 'block1', 'data': 'grass16'},
                {'dir': 'block2', 'data': 'grass32'},],
               '0119_050340')
    n.load_graph('0119_123159')
    f = n.freeze(n.train_program[-1])
    
    #nx.drawing.nx_agraph.write_dot(n.G, './test.dot')
    null = np.zeros([1000, 3, 32, 32])
    t = Tessellater(None, 'grass32', f.name, f.config,
                    32, 256, [1024, 1024], 'grass_gen{}.jpg', 'testing',
                    {'frozen_res_cg_ebm_0/res_alphas_0/alpha0:0': 1.0,
                     'frozen_res_cg_ebm_0/res_alphas_0/alpha1:0': 1.0,
                     'frozen_res_cg_ebm_0/res_alphas_0/alpha2:0': 1.0,
                     #'frozen_res_cg_ebm_0/cqs_concat_0/data_input:0': null,
                     #'frozen_res_cg_ebm_0/cqs_loss_set_1/orig_input:0': null,
                    },
                    f)
    t.set_zinput('frozen_res_cg_ebm_0/res_gen_pair_00/input:0')
    t.set_rinput('frozen_res_cg_ebm_0/res_bridge_1/input:0')
    t.set_zoutput('frozen_res_cg_ebm_0/res_bridge_0/output:0')
    t.set_routput('frozen_res_cg_ebm_0/res_bridge_0/output2:0')

    return n, f, t
