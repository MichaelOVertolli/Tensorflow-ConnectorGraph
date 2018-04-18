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

import gc
from models.graphs.converter import *
from models.graphs.res_cg_ebm_mad import *
from models.configs.res_cg_ebm_mad import *
from models.model_utils import strip_index
from netgenrunner import *
import networkx as nx
from networkx.algorithms.dag import ancestors, descendants


def test():
    nr = NetGenRunner('res_cg_ebm_mad',
                     'scaled_began_gmsm_b16_z128_sz64_h128_g0.7_elu_pconv_wxav_alphas',
                      'celebc/branching/NETGEN_frozen_res_cg_ebm_mad_scaled_began_gmsm_b16_z128_sz64_h128_g0.7_elu_pconv_wxav_alphas_0320_122853/base_block',
                     [
                         'celebc/branching/NETGEN_frozen_res_cg_ebm_mad_scaled_began_gmsm_b16_z128_sz64_h128_g0.7_elu_pconv_wxav_alphas_0320_122853/partial_base_block.gpickle',
                     ],
                      branching=True)

    G = nr.Gs.items()[0][1]
    nr.prep_sess(G, 'celebcr', 1000, [16, 16])

    # goutp = G.graph['gen_outputs']['G'][0]
    # rinpt = G.graph['rev_inputs'][goutp]
    # routp = G.graph['gen_outputs']['R'][0]
    # print G.graph['gen_input']
    # print goutp
    # print rinpt
    # print routp
    # print G.graph['data_inputs']
    # print G.graph['a_output']
    # gout = nr.generate(G.graph['gen_input'], goutp, rinpt, 10)
    # rout = nr.reverse(routp, rinpt, gout)
    # aout = nr.autoencode(G.graph['a_output'], G.graph['data_inputs'], gout)
    # nr.save_img(gout[0], 'netgenrunner_mad_g.jpg')
    # nr.save_img(rout[0], 'netgenrunner_mad_r.jpg')
    # nr.save_img(aout[0], 'netgenrunner_mad_a.jpg')
    losses, tags = nr.run_losses(G, 16)
    print 'Losses done.'
    print tags
    tags = None
    # imgs = pairs[0]
    # nr.save_img(imgs[0], 'netgenrunner_mad_lss_r0.jpg')
    # nr.save_img(imgs[1], 'netgenrunner_mad_lss_r1.jpg')
    # nr.save_img(imgs[2], 'netgenrunner_mad_lss_o.jpg')
    # nr.save_img(imgs[2], 'netgenrunner_mad_lss_a.jpg')
    subsets = nr.subset_data(losses, 0.9)
    # nr.close_sess()

    return nr, losses, subsets
