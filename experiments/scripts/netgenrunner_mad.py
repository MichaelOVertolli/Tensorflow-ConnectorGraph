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

from models.graphs.converter import *
from models.graphs.res_cg_ebm_mad import *
from models.configs.res_cg_ebm_mad import *
from models.model_utils import strip_index
from netgenrunner import *
import networkx as nx
from networkx.algorithms.dag import ancestors, descendants


def test():
    nr = NetGenRunner('res_cg_ebm_mad_fw',
                     'scaled_began_gmsm_b16_z128_sz64_h128_g0.7_elu_pconv_wxav_alphas',
                      'celebc/branching/NETGEN_frozen_res_cg_ebm_mad_scaled_began_gmsm_b16_z128_sz64_h128_g0.7_elu_pconv_wxav_alphas_0319_180009/base_block',
                     [
                         'celebc/branching/NETGEN_res_cg_ebm_mad_fw_scaled_began_gmsm_b16_z128_sz64_h128_g0.7_elu_pconv_wxav_alphas_0319_180009/partial_res_gen_pair_0001_graph.gpickle',
                     ],
                      branching=True)

    G = nr.Gs.items()[0][1]
    nr.prep_sess(G, 'celebc16/splits/train')

    print G.graph['gen_input']
    print G.graph['gen_outputs']['G']
    print G.graph['rev_input']
    print G.graph['data_inputs']
    print G.graph['a_output']
    gout = nr.generate(G.graph['gen_input'], G.graph['gen_outputs']['G'], [G.graph['rev_input']], 10)
    rout = nr.reverse(G.graph['gen_outputs']['R'], G.graph['rev_input'], gout)
    aout = nr.autoencode(G.graph['a_output'], G.graph['data_inputs'], gout)
    nr.save_img(gout[0], 'netgenrunner_full_g.jpg')
    nr.save_img(rout[0], 'netgenrunner_full_r.jpg')
    nr.save_img(aout[0], 'netgenrunner_full_a.jpg')
    nr.close_sess()

    return nr
